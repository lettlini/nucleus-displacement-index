from typing import Callable, Optional

import cv2
import numpy as np
import shapely


def translation(translation_vec: np.array) -> Callable[[np.array], np.array]:

    def translation_x(x: np.array) -> np.array:
        return x + translation_vec

    return translation_x


def rotation(angle: float) -> Callable[[np.array], np.array]:

    def rotate_x(x: np.array) -> np.array:
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return np.dot(x, rotation_matrix.T)

    return rotate_x


def calculate_nusdi_single_frame(
    cell_label_image: np.array,
    nuclei_label_image: np.array,
    dilate: Optional[bool] = False,
) -> dict:
    """ """
    nusdi_dict = {}

    cell_label_image = cell_label_image.copy()
    nuclei_label_image = nuclei_label_image.copy()

    cell_labels = np.setdiff1d(cell_label_image, (0,))
    nucleus_labels = np.setdiff1d(nuclei_label_image, (0,))

    assert len(cell_labels) == len(nucleus_labels)

    if len(cell_labels) != len(nucleus_labels):
        raise RuntimeError(
            f"Number of cells ({len(cell_labels)}) and nuclei ({len(nucleus_labels)}) do not match."
        )

    for current_cell_label in cell_labels:
        cell_selection_mask = cell_label_image == current_cell_label

        corresponding_nucleus = np.setdiff1d(
            nuclei_label_image[cell_selection_mask], (0,)
        ).item()

        nucleus_selection_mask = nuclei_label_image == corresponding_nucleus

        nusdi_dict[current_cell_label] = calculate_nusdi_single_cell(
            cell_selection_mask, nucleus_selection_mask, dilate=dilate
        )

    return nusdi_dict


def get_valid_distance_distribution(cell_polygon, nucleus_polygon, bbox_w, bbox_h):
    valid_distances = []
    ccentroid = np.array(cell_polygon.centroid.xy)

    for dx in np.linspace(-1, 1, 51):
        for dy in np.linspace(-1, 1, 51):
            for alpha in np.linspace(0, 2 * np.pi, 10):
                # transform nucleus_polygon into configuration:
                current_nucleus_polygon = shapely.transform(
                    nucleus_polygon,
                    translation(0.5 * np.array([dx * bbox_w, dy * bbox_h])),
                )
                current_nucleus_polygon = shapely.transform(
                    current_nucleus_polygon, rotation(alpha)
                )
                # check if contained
                if cell_polygon.contains(current_nucleus_polygon):
                    valid_distances.append(
                        np.linalg.norm(
                            np.array(current_nucleus_polygon.centroid.xy) - ccentroid
                        )
                    )

    valid_distances = np.array(valid_distances)

    return valid_distances


def calculate_nusdi_single_cell(
    cell_selection_mask: np.array,
    nucleus_selection_mask: np.array,
    dilate: Optional[bool] = False,
) -> np.array:
    """ """
    if dilate:
        cell_selection_mask = cv2.dilate(
            cell_selection_mask.copy().astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

    cell_points, _ = cv2.findContours(
        cell_selection_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    nucleus_points, _ = cv2.findContours(
        nucleus_selection_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if len(cell_points) != 1:
        raise RuntimeError(
            f"Found {len(cell_points)} cell contours, required exactly 1."
        )

    if len(nucleus_points) != 1:
        raise RuntimeError(
            f"Found {len(nucleus_points)} nucleus contours, required exactly 1."
        )

    cell_points = cell_points[0].squeeze()
    cell_moments = cv2.moments(cell_points)

    nucleus_points = nucleus_points[0].squeeze()
    nucleus_moments = cv2.moments(nucleus_points)

    _, _, bbox_w, bbox_h = cv2.boundingRect(
        cell_points
    )  # coordinates of top left + w / h

    initial_nucleus_centroid = np.array(
        [
            nucleus_moments["m10"] / nucleus_moments["m00"],
            nucleus_moments["m01"] / nucleus_moments["m00"],
        ]
    )
    initial_cell_centroid = np.array(
        [
            cell_moments["m10"] / cell_moments["m00"],
            cell_moments["m01"] / cell_moments["m00"],
        ]
    )

    initial_displacement_vector = initial_nucleus_centroid - initial_cell_centroid

    empirical_distance = np.linalg.norm(initial_displacement_vector)

    # the cell will never change, we only have to init this once
    cell_polygon = shapely.geometry.Polygon(cell_points)
    nucleus_polygon = shapely.geometry.Polygon(nucleus_points)

    # before we start the calculation, we trasform the nucleus into the centroid of the cell
    nucleus_polygon = shapely.transform(
        nucleus_polygon, translation(-initial_displacement_vector)
    )

    distance_distribution = get_valid_distance_distribution(
        cell_polygon, nucleus_polygon, bbox_w, bbox_h
    )

    return calculate_nusdi(distance_distribution, empirical_distance)


def calculate_nusdi(distances: np.array, empirical_distance: float) -> float:

    if len(distances) == 0:
        if empirical_distance > 0:
            return 1
        return 0

    maximum_valid = max(np.max(distances), empirical_distance)

    if maximum_valid == 0:
        return 0

    return empirical_distance / maximum_valid
