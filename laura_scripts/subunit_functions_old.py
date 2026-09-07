# Functions for SNMF subunit plotting

# %% Multipanel figure where each subplot is the SNMF subunits of a top QI cell, with X number of neighbouring cells overlayed (defined by function input)


def plot_top_cells_overlay_snmf_old(dataset, number_of_overlaps):
    # dataset is e.g.r"F:\Laura\zebrafish_02_12_2025\Phase_01\4px_20Hz_40mins_shuffle_idx_4"
    # number of overlaps is how many cells you want to overlay

    # Import dependencies
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import pandas as pd
    from pathlib import Path

    # Load dataset & quality
    root = Path(dataset)
    quality = np.load(root / "quality.npy")

    # Find top cells (QI > 20)
    qi_limit = 20
    valid_rows = ~np.isnan(quality[:, 0])
    top_cells_idx = np.where(quality[valid_rows, 0] > qi_limit)
    top_cells_idx = np.where(valid_rows)[0][top_cells_idx]
    top_cells = quality[top_cells_idx, 1].astype(int)
    top_cells = top_cells[top_cells != 183]  # remove cell 183 (SNMF didn't process)

    # Set the correct number of subplots in the figure
    n_cols = math.ceil(math.sqrt(len(top_cells)))
    n_rows = math.ceil(len(top_cells) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi=300)
    axes = axes.flatten()

    # Set the colours of the overlaying cells
    overlay_colours = ["red", "darkgreen"]

    # For each main cell
    for idx, cell_id in enumerate(top_cells):
        # Load MSE snippet (for the overlap calculation)
        mse_image = np.load(root / f"cell_{cell_id}/snippet_mse.npy")
        h, w = mse_image.shape  # extract dimensions of the mse snippet

        # Extract the actual coordinates of the centre of the cell's receptive field
        x_center = quality[cell_id, 2]
        y_center = quality[cell_id, 3]

        # Plot the subunits of the main cell (in dark blue)
        contours = np.load(
            root / f"cell_{cell_id}/s_nmf_contours.npy", allow_pickle=True
        )

        for contour in contours:
            contour = np.asarray(contour)
            axes[idx].plot(
                contour[:, 1], contour[:, 0], color="darkblue", linewidth=0.8
            )

        # Find which cells overlap the most
        overlaps = []  # create overlaps list to append to later

        for other_cell in top_cells:  # Go through all the other top cells
            if other_cell == cell_id:
                continue  # if the "other cell" is actually the main cell, then just continue as overlap will be 100%
            # load the MSE image of the other cells
            other_mse = np.load(root / f"cell_{other_cell}/snippet_mse.npy")
            # extract the coordinates of the centre of the receptive field of the other cell
            other_x = quality[other_cell, 2]
            other_y = quality[other_cell, 3]
            # Work out how far away the other cell is from the main cell
            diff_x = other_x - x_center
            diff_y = other_y - y_center
            # Work out the coordinates of the other cell compared to the main cell
            y_start = int(max(0, -diff_y))
            y_end = int(min(h, other_mse.shape[0] - diff_y))
            x_start = int(max(0, -diff_x))
            x_end = int(min(w, other_mse.shape[1] - diff_x))
            # From the above, calculate by how much the other cell overlaps with the main cell
            overlap_area = max(0, y_end - y_start) * max(0, x_end - x_start)
            # If the overlap area is greater than 0, then append this cell and its information to the overlaps list
            if overlap_area > 0:
                overlaps.append(
                    {
                        "cell_id": other_cell,
                        "overlap_area": overlap_area,
                        "diff_x": diff_x,
                        "diff_y": diff_y,
                    }
                )

        # Now overlay the subunits for the top N overlapping cells
        if overlaps:  # Convert overlaps to a pandas dataframe and sort by overlap area
            overlaps_df = (
                pd.DataFrame(overlaps)
                .sort_values("overlap_area", ascending=False)
                .reset_index(drop=True)
            )
            # For every overlapping cell (minimum possible is defined in the function, and maximum possible is defined by max number of overlapping cells)
            for i in range(min(number_of_overlaps, len(overlaps_df))):
                overlay_id = overlaps_df.loc[
                    i, "cell_id"
                ]  # find the cell id of the first overlapping cell
                dx = overlaps_df.loc[i, "diff_x"]  # coordinate shift required
                dy = overlaps_df.loc[i, "diff_y"]
                colour = overlay_colours[
                    i % len(overlay_colours)
                ]  # colour of overlapping cell

                overlay_contours = np.load(
                    root / f"cell_{overlay_id}/s_nmf_contours.npy", allow_pickle=True
                )  # load the subunits of the overlapping cell

                for contour in overlay_contours:  # plot the subunits of overlaying cell
                    contour = np.asarray(contour)
                    axes[idx].plot(
                        contour[:, 1] + dx,
                        contour[:, 0] + dy,
                        color=colour,
                        linewidth=0.6,
                        alpha=0.7,
                    )

            top_overlap_ids = (
                overlaps_df["cell_id"].iloc[:number_of_overlaps].tolist()
            )  # add the id of the overlapping cells to the top_overlaps_id list (for future subtitle)
        else:
            top_overlap_ids = []  # if no overlapping cells, the list is empty

        # Subplot titles
        if (
            top_overlap_ids
        ):  # if there are overlapping cells present, add them to the subtitle
            axes[idx].set_title(
                f"Cell {cell_id} ({', '.join(map(str, top_overlap_ids))})", fontsize=10
            )
        else:  # otherwise just put the main cell in the subtitle
            axes[idx].set_title(f"Cell {cell_id}", fontsize=10)

        # Subplot axes
        axes[idx].set_xlim(0, w)  # 60 by 60 pixels
        axes[idx].set_ylim(h, 0)
        # axes[idx].set_aspect("equal")
        axes[idx].set_xticks([0, w / 2, w])
        axes[idx].set_yticks([0, h / 2, h])
        axes[idx].set_xticklabels([0, w * 2, w * 4])
        axes[idx].set_yticklabels([0, h * 2, h * 4])
        axes[idx].set_xlabel("µm")
        axes[idx].set_ylim(axes[idx].get_ylim()[::-1])

        # Add scale bar (each pixel is 4 micrometres)
        bar_length_px = 2
        x_end_bar = w - 5
        x_start_bar = x_end_bar - bar_length_px
        y_pos = 8

        axes[idx].plot(
            [x_start_bar, x_end_bar], [y_pos, y_pos], color="red", linewidth=3
        )  # plot scale bar
        axes[idx].text(
            x_start_bar - 5, y_pos - 4, "8 µm", color="red", fontsize=9
        )  # plot scale bar label

    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    # Set overall figure title
    fig.suptitle(
        f"SNMF subunit receptive fields (QI > 20) with top {number_of_overlaps} overlapping cells\n",
        fontsize=20,
        y=0.99,
    )

    # Plot figure
    plt.tight_layout()
    plt.show()


# %% Function to plot the subunits of a single main cell in the first subplot, and then add progressive subplots for the top spatially overlapping cells


def plot_single_cell_overlays_snmf_old(dataset, main_cell_id, number_of_overlays=2):
    # dataset is e.g.r"F:\Laura\zebrafish_02_12_2025\Phase_01\4px_20Hz_40mins_shuffle_idx_4"
    # main cell id is the main cell you want to plot, should be one of the top QI > 20 cells
    # number of overlaps is how many cells you want to overlay

    # Import dependencies
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path
    import math

    # Load dataset and quality
    root = Path(dataset)
    quality = np.load(root / "quality.npy")

    # Find top cells (QI > 20)
    qi_limit = 20
    valid_rows = ~np.isnan(quality[:, 0])
    top_cells_idx = np.where(quality[valid_rows, 0] > qi_limit)
    top_cells_idx = np.where(valid_rows)[0][top_cells_idx]
    top_cells = quality[top_cells_idx, 1].astype(int)
    top_cells = top_cells[top_cells != 183]  # remove cell 183 (SNMF didn't process)

    # Load main cell data
    # Load receptive field (MSE snippet) for the main cell
    mse_main = np.load(root / f"cell_{main_cell_id}/snippet_mse.npy")
    h, w = mse_main.shape  # extract dimensions of MSE snippet
    x_center = quality[
        main_cell_id, 2
    ]  # extract coordinates of centre of receptive field for main cell
    y_center = quality[main_cell_id, 3]
    # Load subunit contours (SNMF) for main cell
    main_contours = np.load(
        root / f"cell_{main_cell_id}/s_nmf_contours.npy", allow_pickle=True
    )

    # Find which cells overlap the most
    overlaps = []  # create overlaps list to append to later

    for other_cell in top_cells:  # Go through all the other top cells
        if other_cell == main_cell_id:
            continue  # if the "other cell" is actually the main cell, then just continue as overlap will be 100%
        # load the MSE image of the other cells
        other_mse = np.load(root / f"cell_{other_cell}/snippet_mse.npy")
        # extract the coordinates of the centre of the receptive field of the other cell
        other_x = quality[other_cell, 2]
        other_y = quality[other_cell, 3]
        # Work out how far away the other cell is from the main cell
        diff_x = other_x - x_center
        diff_y = other_y - y_center
        # Work out the coordinates of the other cell compared to the main cell
        y_start = int(max(0, -diff_y))
        y_end = int(min(h, other_mse.shape[0] - diff_y))
        x_start = int(max(0, -diff_x))
        x_end = int(min(w, other_mse.shape[1] - diff_x))
        # From the above, calculate by how much the other cell overlaps with the main cell
        overlap_area = max(0, y_end - y_start) * max(0, x_end - x_start)
        # If the overlap area is greater than 0, then append this cell and its information to the overlaps list
        if overlap_area > 0:
            overlaps.append(
                {
                    "cell_id": other_cell,
                    "overlap_area": overlap_area,
                    "diff_x": diff_x,
                    "diff_y": diff_y,
                }
            )

    overlaps_df = (  # Convert overlaps to a pandas dataframe and sort by overlap area
        pd.DataFrame(overlaps)
        .sort_values("overlap_area", ascending=False)
        .reset_index(drop=True)
        .iloc[:number_of_overlays]
    )

    # Set figure subplot layout
    n_panels = number_of_overlays + 1  # set the number of subplot panels
    n_cols = math.ceil(math.sqrt(n_panels))  # find appropriate number of cols
    n_rows = math.ceil(n_panels / n_cols)  # find appropriate number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=300)
    axes = axes.flatten()

    # Set plotting colours of overlaying cells. Add more here if you want to plot more than 5 overlapping cells
    overlay_colours = ["red", "darkgreen", "purple", "darkorange", "cyan"]

    # Plot the subplots
    for panel_idx in range(n_panels):
        # Plot the main cell subunits in the first panel
        for contour in main_contours:
            contour = np.asarray(contour)
            axes[panel_idx].plot(
                contour[:, 1], contour[:, 0], color="darkblue", linewidth=0.9
            )

        # Add overlays progressively
        for i in range(
            panel_idx
        ):  # For as many subplots as you have, plot the equivalent number of overlapping cells -1
            overlay_id = overlaps_df.loc[
                i, "cell_id"
            ]  # load the id of the overlapping cell
            dx = overlaps_df.loc[
                i, "diff_x"
            ]  # load the shifting coordinates of the overlapping cell
            dy = overlaps_df.loc[i, "diff_y"]
            colour = overlay_colours[
                i % len(overlay_colours)
            ]  # load the plotting colour of the overlapping cell
            overlay_contours = np.load(
                root / f"cell_{overlay_id}/s_nmf_contours.npy", allow_pickle=True
            )  # Load the subunit contours of the overlaying cell

            # Plot the overlapping cell subunits
            for contour in overlay_contours:
                contour = np.asarray(contour)
                axes[panel_idx].plot(
                    contour[:, 1] + dx,
                    contour[:, 0] + dy,
                    color=colour,
                    linewidth=0.6,
                    alpha=0.7,
                )

        # Set subplot titles
        if panel_idx == 0:  # for the first panel, set the subtitle to "Cell X"
            axes[panel_idx].set_title(f"Cell {main_cell_id}", fontsize=10)
        else:  # for every other panel, set it to "Cell X + Y, Z etc"
            ids = overlaps_df["cell_id"].iloc[:panel_idx].tolist()
            axes[panel_idx].set_title(
                f"Cell {main_cell_id} + {', '.join(map(str, ids))}", fontsize=10
            )

        # Set axes
        axes[panel_idx].set_xlim(0, w)
        axes[panel_idx].set_ylim(h, 0)
        axes[panel_idx].set_xticks([0, w / 2, w])
        axes[panel_idx].set_yticks([0, h / 2, h])
        axes[panel_idx].set_xticklabels([0, w * 2, w * 4])
        axes[panel_idx].set_yticklabels([0, h * 2, h * 4])
        axes[panel_idx].set_xlabel("µm")
        axes[panel_idx].set_ylabel("µm")
        axes[panel_idx].set_ylim(axes[panel_idx].get_ylim()[::-1])

        # Add scale bar (each pixel is 4 micrometres)
        bar_length_px = 2
        x_end_bar = w - 5
        x_start_bar = x_end_bar - bar_length_px
        y_pos = 8
        axes[panel_idx].plot(
            [x_start_bar, x_end_bar], [y_pos, y_pos], color="red", linewidth=3
        )  # plot scale bar
        axes[panel_idx].text(
            x_start_bar - 3, y_pos - 3, "8 µm", color="red", fontsize=9
        )  # plot scale bar label

    # Remove unused axes
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    # Set overall figure title
    fig.suptitle(
        f"SNMF subunit overlays for cell {main_cell_id}\n", fontsize=16, y=0.98
    )

    # Plot figure
    plt.tight_layout()
    plt.show()


# %% Function to plot SNMF subunits for all top cells back onto MEA visual field coordinates


def plot_subunit_mosaic_old(dataset, pixel_dimensions):
    # dataset is e.g.r"F:\Laura\zebrafish_02_12_2025\Phase_01\4px_20Hz_40mins_shuffle_idx_4"
    # pixel dimensions is e.g. 600 for 600x600 pixels (4px noise), 800 for 800x800 (12px noise)

    # Import dependencies
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from matplotlib import colormaps

    # Load dataset & quality
    root = Path(dataset)
    quality = np.load(root / "quality.npy")

    # Find top cells (QI > 20)
    qi_limit = 20
    valid_rows = ~np.isnan(quality[:, 0])
    top_cells_idx = np.where(quality[valid_rows, 0] > qi_limit)
    top_cells_idx = np.where(valid_rows)[0][top_cells_idx]
    top_cells = quality[top_cells_idx, 1].astype(int)
    top_cells = top_cells[top_cells != 183]  # remove cell 183 (SNMF didn't process)
    cells = list(top_cells)

    # Set figure
    fig, axes = plt.subplots(figsize=(10, 10), dpi=300)
    colours = (
        (colormaps["tab20"].colors)
        + (colormaps["tab20b"].colors)
        + (colormaps["tab20c"].colors)
    )  # creates a list of 60 colours

    for idx, cell in enumerate(cells):
        # Plot background mse receptive fields
        image = np.load(
            root / f"cell_{int(cell)}/snippet_mse.npy"
        )  # Import subset_dev_img.npy for cells
        x_centre = quality[
            cell, 2
        ]  # Find cell's RF's actual position (x, y) from quality array
        y_centre = quality[cell, 3]
        (
            h,
            w,
        ) = image.shape  # receptive field image is 60 x 60 pixels (from mse snippets)
        extent = [
            x_centre - w / 2,
            x_centre + w / 2,
            y_centre - h / 2,
            y_centre + h / 2,
        ]  # calculates x0,x1,y0,y1 coordinates for plotting
        axes.imshow(
            image, cmap="Greys", origin="lower", extent=extent, alpha=0.5
        )  # plot mse receptive field in greyscale

        # Plot subunits on top
        subunit_root = root / f"cell_{cell}"
        contours = np.load(
            subunit_root / "s_nmf_contours.npy", allow_pickle=True
        )  # (pickle can allow handling of non-numbers in np array...)
        axes.plot(
            [], [], color=colours[idx], label=f"Cell {cell}"
        )  # add colour line (with no data) per cell for legend

        for contour in contours:
            contour = np.asarray(contour)
            axes.plot(
                contour[:, 1] - w / 2 + x_centre + 0.5,
                contour[:, 0] - h / 2 + y_centre + 0.5,
                color=colours[idx],
                linewidth=0.4,
            )
            # ^^ SHIFT BY 0.5 pixels is required in both x and y directions since there is no "centre" coordinate in a 60x60 pixel array.

    # Plot ticks and labels
    axes.set_ylim([0, pixel_dimensions])  # 600 pixels for 4px noise, 800 for 12px noise
    axes.set_xlim([0, pixel_dimensions])
    axes.set_yticks(np.arange(0, pixel_dimensions + 1, 100))
    axes.set_xticks(np.arange(0, pixel_dimensions + 1, 100))
    axes.set_xticklabels(np.arange(0, (pixel_dimensions * 4) + 1, 400))
    axes.set_yticklabels(np.arange(0, (pixel_dimensions * 4) + 1, 400))
    axes.set_xlabel("µm")
    axes.set_ylabel("µm")
    axes.legend(frameon=False)

    # Add scale bar, 8 µm = 2 pixels
    bar_length_px = 2
    x_start_bar = 570
    x_end_bar = x_start_bar + bar_length_px
    y_pos = 570
    axes.plot([x_start_bar, x_end_bar], [y_pos, y_pos], color="red", linewidth=3)
    axes.text(x_start_bar - 13, y_pos + 11, "8 µm", color="red", fontsize=15)
    axes.legend(fontsize=5, frameon=False)

    fig.suptitle(
        "Subunit contours (SNMF) for cells with QI > 20 (white 4px_20Hz_shuffle noise)"
    )
    fig.tight_layout()
    fig.show()


# %%
