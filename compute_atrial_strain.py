import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math
import warnings
from scipy.spatial import distance
from skimage import measure
import cv2
from collections import defaultdict
from numpy.linalg import norm
from scipy.interpolate import splprep, splev, splrep
from scipy.ndimage.morphology import binary_closing as closing
from skimage.morphology import skeletonize
from collections import Counter
from skimage.measure import label
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import resample
import numpy as np
debug = False
Nsegments_length = 15

# Enable interactive mode in matplotlib
# plt.ion()


def smooth_curve_1d(curve, num_frames, s, num_points=None, x_values=None):
    """Smooth via spline interpolation, with smoothing condition s
    Use num_points to increase number of points in curve
    If changing number of points, also change x axis values to match"""
    x = np.linspace(0, num_frames - 1, num_frames)

    if num_points is not None:
        xx = np.linspace(x.min(), x.max(), num_points)
        curve_interp = interp1d(x, curve, kind='linear')(xx)
        x_interp = interp1d(x, x_values, kind='linear')(xx)
        spl = splrep(xx, curve_interp, s=s)
        curve_smooth = splev(xx, spl)
        return curve_smooth, x_interp
    else:
        spl = splrep(x, curve, s=s)
        curve_smooth = splev(x, spl)
        return curve_smooth


def getLargestCC(segmentation):
    nb_labels = np.unique(segmentation)[1:]
    out_image = np.zeros_like(segmentation)

    # Loop over labels, leaving myocardium for last
    for ncc in [x for x in nb_labels if x != 2]:
        _aux = np.squeeze(segmentation == ncc).astype(float)  
        labels = label(_aux)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        out_image += largestCC * ncc

    # Handle myocardium last in case of overlaps
    if 2 in nb_labels:
        ncc = 2
        _aux = np.squeeze(segmentation == ncc).astype(float)  
        labels = label(_aux)

        # If myocardium has multiple connected components, dilate to connect components
        if (1.0 and 2.0 in np.unique(segmentation)) and np.unique(labels).shape[0] > 2:
            kernel = np.ones((2, 2), np.uint8)
            _aux = cv2.dilate(_aux, kernel, iterations=2)
            cnt_myo_seg_dil = measure.find_contours(_aux, 0.8)
            cnt_myo_seg = measure.find_contours(_aux, 0.8)
            if len(cnt_myo_seg_dil) > 1 and len(cnt_myo_seg) > 1:
                _aux = dilate_LV_myo(segmentation)
            labels = label(_aux)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

        # If myo overlaps other labels, it will overwrite
        out_image[largestCC] = 2

    return out_image


def dilate_LV_myo(seg):
    # Load the image array with the three masks
    try:
        myo_seg = np.squeeze(seg == 2).astype(float)
        bp_seg = np.squeeze(seg == 1).astype(int)
        la_seg = np.squeeze(seg == 3).astype(int)
        # find the indices of the pixels in structure1 and structure2
        idx_structure1 = np.argwhere(bp_seg == 1)
        idx_structure2 = np.argwhere(la_seg == 1)
        # compute the distance between each pixel in structure1 and the closest pixel in structure2
        # using the Euclidean distance
        distances_x = distance.cdist(
            idx_structure1, idx_structure2, 'euclidean').min(axis=1)
        # Get the threshold value for the 25% furthest pixels
        threshold = np.percentile(distances_x, 75)
        # create a new array with the same shape as the original mask
        dist_map = np.zeros_like(bp_seg, dtype=float)
        dist_map[idx_structure1[:, 0], idx_structure1[:, 1]] = distances_x
        structure9 = np.zeros_like(bp_seg)

        # Set pixels that are 25% furthest away from the center of structure2 to 1
        structure9[np.where(dist_map > threshold)] = 1
        structure9 = structure9.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        structure9_dil = cv2.dilate(structure9, kernel, iterations=2)
        combined_mask = cv2.bitwise_or(
            myo_seg.astype(np.uint8), structure9_dil)
        # Set pixels to zero where mask3 has value 1
        combined_mask[np.where(bp_seg == 1)] = 0
        if debug:
            plt.imshow(myo_seg)
            plt.imshow(combined_mask)
    except:
        combined_mask = seg

    return combined_mask


def binarymatrix(A):
    A_aux = np.copy(A)
    A = map(tuple, A)
    dic = Counter(A)
    for (i, j) in dic.items():
        if j > 1:
            ind = np.where(((A_aux[:, 0] == i[0]) & (A_aux[:, 1] == i[1])))[0]
            A_aux = np.delete(A_aux, ind[1:], axis=0)
    if np.linalg.norm(A_aux[:, 0] - A_aux[:, -1]) < 0.01:
        A_aux = A_aux[:-1, :]
    return A_aux


# Define a function to calculate the Euclidean distance between two points
def find_distance_x(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_atrial_circumference(contours, atria_edge1, atria_edge2, top_atrium, fr, study_ID):
    # coordinates of the non-linear line (example)
    indice1 = np.where((contours == (sorted(contours, key=lambda p: find_distance_x(
        p[0], p[1], atria_edge1[0], atria_edge1[1]))[0])).all(axis=1))[0][0]
    indice2 = np.where((contours == (sorted(contours, key=lambda p: find_distance_x(
        p[0], p[1], atria_edge2[0], atria_edge2[1]))[0])).all(axis=1))[0][0]
    indice3 = np.where((contours == (sorted(contours, key=lambda p: find_distance_x(
        p[0], p[1], top_atrium[0], top_atrium[1]))[0])).all(axis=1))[0][0]
    
    # deal with cases where contour goes on from end to start of array.
    if indice1 < indice2:
        if indice1 < indice3 < indice2:
            contour_atria = contours[indice1:indice2+1, :]
        else:
            contour_atria = np.concatenate(
                (contours[indice2:, :], contours[:indice1+1, :]), axis=0
            )
    else:
        if indice2 < indice3 < indice1:
            contour_atria = contours[indice2:indice1+1, :]
        else:
            contour_atria = np.concatenate(
                (contours[indice1:, :], contours[:indice2+1, :]), axis=0
            )

    # calculate the length of the line using the distance formula
    total_length = 0
    if debug:
        plt.figure()
        plt.plot(contours[:, 1], contours[:, 0], 'r-')
        plt.plot(contour_atria[:, 1], contour_atria[:, 0], label='atrium contour')
        plt.plot(atria_edge1[1], atria_edge1[0], 'go', label='atrium edge 1')
        plt.plot(atria_edge2[1], atria_edge2[0], 'bo', label='atrium edge 2')
        plt.plot(top_atrium[1], top_atrium[0], 'mo', label='atrium top')
        plt.legend()
        plt.title('{0} frame {1}'.format(study_ID, fr))

    for i in range(len(contour_atria)-1):
        x1, y1 = contour_atria[i, :]
        x2, y2 = contour_atria[i+1, :]
        segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_length += segment_length
    return total_length


def get_strain(atrial_circumferences_per_phase):
    diff_lengths = atrial_circumferences_per_phase - \
        atrial_circumferences_per_phase[0]
    strain = np.divide(diff_lengths, atrial_circumferences_per_phase[0]) * 100
    return strain


def get_right_atrial_volumes(seg, _fr, _pointsRV, logger):
    """
    This function gets the centre line (height) of the atrium and atrial dimension at 15 points along this line.
    """
    _apex_RV, _rvlv_point, _free_rv_point = _pointsRV
    if debug:
        plt.figure()
        plt.imshow(seg)
        plt.plot(_apex_RV[1], _apex_RV[0], 'mo', label='apex')
        plt.plot(_rvlv_point[1], _rvlv_point[0], 'c*', label='rvlv')
        plt.plot(_free_rv_point[1], _free_rv_point[0], 'y*', label='free_rv')
        plt.legend()

    mid_valve_RV = np.mean([_rvlv_point, _free_rv_point], axis=0)
    _atria_seg = np.squeeze(seg == 5).astype(float)  # get atria label
    rv_seg = np.squeeze(seg == 3).astype(float)  # get atria label

    # Generate contours from the atria
    _contours_RA = measure.find_contours(_atria_seg, 0.8)
    _contours_RA = _contours_RA[0]

    contours_RV = measure.find_contours(rv_seg, 0.8)
    contours_RV = contours_RV[0]

    # Fit a spline to smooth out the atrium
    n_samples = 100
    x = binarymatrix(_contours_RA)
    spl, u = splprep(x.T, s=5, quiet=2, nest=-1)
    u_new = np.linspace(u.min(), u.max(), n_samples)
    _contours_RA = np.zeros([n_samples, 2])
    _contours_RA[:, 0], _contours_RA[:, 1] = splev(u_new, spl)

    # Compute distance between mid_valve and every point in contours
    dist = distance.cdist(_contours_RA, [mid_valve_RV])
    ind_mitral_valve = dist.argmin()
    mid_valve_RA = _contours_RA[ind_mitral_valve, :]
    dist = distance.cdist(_contours_RA, [mid_valve_RA])
    ind_top_atria = dist.argmax()
    top_atria = _contours_RA[ind_top_atria, :]
    ind_base1 = distance.cdist(_contours_RA, [_rvlv_point]).argmin()
    ind_base2 = distance.cdist(_contours_RA, [_free_rv_point]).argmin()
    atria_edge1 = _contours_RA[ind_base1, :]
    atria_edge2 = _contours_RA[ind_base2, :]

    if debug:
        plt.figure()
        plt.imshow(seg)
        plt.plot(_contours_RA[:, 1], _contours_RA[:, 0], 'r-')
        plt.plot(contours_RV[:, 1], contours_RV[:, 0], 'k-')
        plt.plot(top_atria[1], top_atria[0], 'mo', label='top RA')
        plt.plot(mid_valve_RA[1], mid_valve_RA[0], 'co', label='midvalve RA')
        plt.plot(atria_edge1[1], atria_edge1[0], 'go', label='atrium edge 1')
        plt.plot(atria_edge2[1], atria_edge2[0], 'yo', label='atrium edge 2')
        plt.plot(_rvlv_point[1], _rvlv_point[0], 'k*', label='rvlv')
        plt.plot(_free_rv_point[1], _free_rv_point[0], 'b*', label='free rv')
        plt.legend()

    # Rotate contours by theta degrees
    radians = np.arctan2(np.array((atria_edge1[0] - atria_edge2[0]) / 2),
                         np.array((atria_edge1[1] - atria_edge2[1]) / 2))

    # Rotate contours
    _x = _contours_RA[:, 1]
    y = _contours_RA[:, 0]
    xx_B = _x * math.cos(radians) + y * math.sin(radians)
    yy_B = -_x * math.sin(radians) + y * math.cos(radians)

    # Rotate points
    x_1 = atria_edge1[1]
    y_1 = atria_edge1[0]
    x_2 = atria_edge2[1]
    y_2 = atria_edge2[0]
    x_4 = top_atria[1]
    y_4 = top_atria[0]
    x_5 = mid_valve_RA[1]
    y_5 = mid_valve_RA[0]

    xx_1 = x_1 * math.cos(radians) + y_1 * math.sin(radians)
    yy_1 = -x_1 * math.sin(radians) + y_1 * math.cos(radians)
    xx_2 = x_2 * math.cos(radians) + y_2 * math.sin(radians)
    yy_2 = -x_2 * math.sin(radians) + y_2 * math.cos(radians)
    xx_4 = x_4 * math.cos(radians) + y_4 * math.sin(radians)
    yy_4 = -x_4 * math.sin(radians) + y_4 * math.cos(radians)
    xx_5 = x_5 * math.cos(radians) + y_5 * math.sin(radians)
    yy_5 = -x_5 * math.sin(radians) + y_5 * math.cos(radians)

    # make vertical line through mid_valve_from_atriumcontours_rot
    contours_RA_rot = np.asarray([xx_B, yy_B]).T
    top_atria_rot = np.asarray([xx_4, yy_4])

    # Make more points for the contours.
    intpl_XX = []
    intpl_YY = []
    for ind, coords in enumerate(contours_RA_rot):
        coords1 = coords
        if ind < (len(contours_RA_rot) - 1):
            coords2 = contours_RA_rot[ind + 1]

        else:
            coords2 = contours_RA_rot[0]
        warnings.simplefilter('ignore', np.RankWarning)
        coeff = np.polyfit([coords1[0], coords2[0]], [
                           coords1[1], coords2[1]], 1)
        xx_es = np.linspace(coords1[0], coords2[0], 10)
        intp_val = np.polyval(coeff, xx_es)
        intpl_XX = np.hstack([intpl_XX, xx_es])
        intpl_YY = np.hstack([intpl_YY, intp_val])

    contour_smth = np.vstack([intpl_XX, intpl_YY]).T

    # find the crossing between vert_line and contours_RA_rot.
    dist2 = distance.cdist(contour_smth, [top_atria_rot])
    min_dist2 = np.min(dist2)
    # # step_closer
    newy_atra = top_atria_rot[1] + min_dist2
    new_top_atria = [top_atria_rot[0], newy_atra]
    dist3 = distance.cdist(contour_smth, [new_top_atria])
    ind_min_dist3 = dist3.argmin()

    ind_alt_atria_top = contours_RA_rot[:, 1].argmin()
    final_mid_avalve = np.asarray([xx_5, yy_5])
    final_top_atria = np.asarray(
        [contours_RA_rot[ind_alt_atria_top, 0], contours_RA_rot[ind_alt_atria_top, 1]])
    final_perp_top_atria = contour_smth[ind_min_dist3, :]
    final_atrial_edge1 = np.asarray([xx_1, yy_1])
    final_atrial_edge2 = np.asarray([xx_2, yy_2])

    if debug:
        plt.figure()
        plt.plot(contour_smth[:, 0], contour_smth[:, 1], 'r-')
        plt.plot(final_atrial_edge2[0], final_atrial_edge2[1], 'y*', label='atrium edge 2')
        plt.plot(final_atrial_edge1[0], final_atrial_edge1[1], 'm*', label='atrium edge 1')
        plt.plot(final_top_atria[0], final_top_atria[1], 'c*', label='RA top')
        plt.plot(final_mid_avalve[0], final_mid_avalve[1], 'b*', label='midvalve')
        plt.title('RA 4Ch frame {}'.format(_fr))
        plt.legend()

    alength_top = distance.pdist([final_mid_avalve, final_top_atria])[0]
    alength_perp = distance.pdist([final_mid_avalve, final_perp_top_atria])[0]
    a_segmts = (final_mid_avalve[1] - final_top_atria[1]) / Nsegments_length

    # get length dimension (width) of atrial seg at each place.
    a_diams = np.zeros(Nsegments_length)
    diam1 = abs(np.diff([xx_1, xx_2]))
    points_aux = np.zeros(((Nsegments_length - 1) * 2, 2))
    k = 0
    for ib in range(Nsegments_length):
        if ib == 0:
            a_diams[ib] = diam1
        else:
            # Get next vertical y point
            vert_y = final_mid_avalve[1] - a_segmts * ib

            # Find all y points close to vert_y
            rgne_vertY = a_segmts / 6
            min_Y = vert_y - rgne_vertY
            max_Y = vert_y + rgne_vertY
            ind_sel_conts = np.where(np.logical_and(
                intpl_YY >= min_Y, intpl_YY <= max_Y))[0]

            if len(ind_sel_conts) == 0:
                logger.error(f'Problem in frame {_fr} disk {ib} getting y points close to vert_y')
                continue

            # Find coordinates of points close to vert_y
            y_sel_conts = contour_smth[ind_sel_conts, 1]
            x_sel_conts = contour_smth[ind_sel_conts, 0]
            min_ys = np.argmin(np.abs(y_sel_conts - vert_y))
            p1 = ind_sel_conts[min_ys]
            point1 = contour_smth[p1]

            # Mean is roughly the midpoint horizontally
            mean_x = np.mean([np.min(x_sel_conts), np.max(x_sel_conts)])
            if mean_x < point1[0]:  # If point1 is on right side
                ind_xs = np.where(contour_smth[ind_sel_conts, 0] < mean_x)[0]
                pts = contour_smth[ind_sel_conts[ind_xs], :]
                min_ys = np.argmin(np.abs(pts[:, 1] - vert_y))
                point2 = pts[min_ys]
                a_diam = distance.pdist([point1, point2])[0]
            elif np.min(x_sel_conts) == np.max(x_sel_conts):
                # If only points all along same horizontal plane
                logger.info(
                    'Frame {}, disk {} diameter is zero'.format(_fr, ib))
                a_diam = 0
                point2 = np.zeros(2)
                point1 = np.zeros(2)
            else:
                ind_xs = np.where(contour_smth[ind_sel_conts, 0] > mean_x)[0]
                if len(ind_xs) > 0:
                    pts = contour_smth[ind_sel_conts[ind_xs], :]
                    min_ys = np.argmin(np.abs(pts[:, 1] - vert_y))
                    point2 = pts[min_ys]
                    a_diam = distance.pdist([point1, point2])[0]
                else:
                    a_diam = 0
                    point2 = np.zeros(2)
                    point1 = np.zeros(2)
                    logger.info(
                        'ra_4Ch: Frame {}, disk {} diameter is zero'.format(_fr, ib))

            a_diams[ib] = a_diam
            points_aux[k, :] = point1
            points_aux[k + 1, :] = point2

            k += 2

    points_rotate = np.zeros(((Nsegments_length - 1) * 2 + 5, 2))
    points_rotate[0, :] = final_mid_avalve
    points_rotate[1, :] = final_top_atria
    points_rotate[2, :] = final_perp_top_atria
    points_rotate[3, :] = final_atrial_edge1
    points_rotate[4, :] = final_atrial_edge2
    points_rotate[5:, :] = points_aux

    radians2 = 2 * np.pi - radians
    points_non_rotate_ = np.zeros_like(points_rotate)
    for _jj, p in enumerate(points_non_rotate_):
        points_non_rotate_[_jj, 0] = points_rotate[_jj, 0] * math.cos(radians2) + points_rotate[_jj, 1] * math.sin(
            radians2)
        points_non_rotate_[_jj, 1] = -points_rotate[_jj, 0] * math.sin(radians2) + points_rotate[_jj, 1] * math.cos(
            radians2)

    length_apex = distance.pdist([_apex_RV, _free_rv_point])
    if debug:
        plt.close('all')
    return a_diams, alength_top, alength_perp, points_non_rotate_, _contours_RA, length_apex


def get_left_atrial_volumes(seg, _seq, _fr, _points, logger):
    """
    This function gets the centre line (height) of the atrium and atrial dimension at 15 points along this line.
    """
    _apex, _mid_valve, anterior_2Ch, inferior_2Ch = _points
    if debug:
        plt.figure()
        plt.imshow(seg)
        plt.plot(_apex[1], _apex[0], 'mo', label='apex')
        plt.plot(_mid_valve[1], _mid_valve[0], 'c*', label='midvalve')
        plt.plot(anterior_2Ch[1], anterior_2Ch[0], 'y*', label='anterior')
        plt.plot(inferior_2Ch[1], inferior_2Ch[0], 'r*', label='inferior')
        plt.legend()

    if _seq == 'la_2Ch':
        _atria_seg = np.squeeze(seg == 3).astype(float)  # get atria label
    else:
        _atria_seg = np.squeeze(seg == 4).astype(float)  # get atria label

    # Generate contours from the atria
    contours = measure.find_contours(_atria_seg, 0.8)
    contours = contours[0]

    # Fit a spline to smooth out the atrium
    n_samples = contours.shape[0]
    x = binarymatrix(contours)
    spl, u = splprep(x.T, s=5, quiet=2, nest=-1)
    u_new = np.linspace(u.min(), u.max(), n_samples)
    contours = np.zeros([n_samples, 2])
    contours[:, 0], contours[:, 1] = splev(u_new, spl)

    # Compute distance between mid_valve and every point in contours
    dist = distance.cdist(contours, [_mid_valve])
    ind_mitral_valve = dist.argmin()
    _mid_valve = contours[ind_mitral_valve, :]
    dist = distance.cdist(contours, [contours[ind_mitral_valve, :]])
    ind_top_atria = dist.argmax()
    top_atria = contours[ind_top_atria, :]
    length_apex_mid_valve = distance.pdist([_apex, _mid_valve])
    length_apex_inferior_2Ch = distance.pdist([_apex, inferior_2Ch])
    length_apex_anterior_2Ch = distance.pdist([_apex, anterior_2Ch])
    lines_LV_ = np.concatenate(
        [length_apex_mid_valve, length_apex_inferior_2Ch, length_apex_anterior_2Ch])
    points_LV_ = np.vstack([_apex, _mid_valve, inferior_2Ch, anterior_2Ch])

    ind_base1 = distance.cdist(contours, [inferior_2Ch]).argmin()
    ind_base2 = distance.cdist(contours, [anterior_2Ch]).argmin()
    atria_edge1 = contours[ind_base1, :]
    atria_edge2 = contours[ind_base2, :]
    # mid valve based on atria
    x_mid_valve_atria = atria_edge1[0] + \
        ((atria_edge2[0] - atria_edge1[0]) / 2)
    y_mid_valve_atria = atria_edge1[1] + \
        ((atria_edge2[1] - atria_edge1[1]) / 2)
    mid_valve_atria = np.array([x_mid_valve_atria, y_mid_valve_atria])
    ind_mid_valve = distance.cdist(contours, [mid_valve_atria]).argmin()
    mid_valve_atria = contours[ind_mid_valve, :]

    if debug:
        plt.figure()
        plt.imshow(seg)
        plt.plot(top_atria[1], top_atria[0], 'mo', label='LA top')
        plt.plot(mid_valve_atria[1], mid_valve_atria[0], 'c*', label='midvalve')
        plt.plot(atria_edge1[1], atria_edge1[0], 'y*', label='atrium edge 1')
        plt.plot(atria_edge2[1], atria_edge2[0], 'r*', label='atrium edge 2')
        plt.legend()

    # Rotate contours by theta degrees
    radians = np.arctan2(np.array((atria_edge2[0] - atria_edge1[0]) / 2),
                         np.array((atria_edge2[1] - atria_edge1[1]) / 2))

    # Rotate contours
    _x = contours[:, 1]
    y = contours[:, 0]
    xx_B = _x * math.cos(radians) + y * math.sin(radians)
    yy_B = -_x * math.sin(radians) + y * math.cos(radians)

    # Rotate points
    x_1 = atria_edge1[1]
    y_1 = atria_edge1[0]
    x_2 = atria_edge2[1]
    y_2 = atria_edge2[0]
    x_4 = top_atria[1]
    y_4 = top_atria[0]
    x_5 = mid_valve_atria[1]
    y_5 = mid_valve_atria[0]

    xx_1 = x_1 * math.cos(radians) + y_1 * math.sin(radians)
    yy_1 = -x_1 * math.sin(radians) + y_1 * math.cos(radians)
    xx_2 = x_2 * math.cos(radians) + y_2 * math.sin(radians)
    yy_2 = -x_2 * math.sin(radians) + y_2 * math.cos(radians)
    xx_4 = x_4 * math.cos(radians) + y_4 * math.sin(radians)
    yy_4 = -x_4 * math.sin(radians) + y_4 * math.cos(radians)
    xx_5 = x_5 * math.cos(radians) + y_5 * math.sin(radians)
    yy_5 = -x_5 * math.sin(radians) + y_5 * math.cos(radians)

    # make vertical line through mid_valve_from_atrium
    contours_rot = np.asarray([xx_B, yy_B]).T
    top_atria_rot = np.asarray([xx_4, yy_4])

    # Make more points for the contours.
    intpl_XX = []
    intpl_YY = []
    for ind, coords in enumerate(contours_rot):
        coords1 = coords
        if ind < (len(contours_rot) - 1):
            coords2 = contours_rot[ind + 1]
        else:
            coords2 = contours_rot[0]
        warnings.simplefilter('ignore', np.RankWarning)
        coeff = np.polyfit([coords1[0], coords2[0]], [
                           coords1[1], coords2[1]], 1)
        xx_es = np.linspace(coords1[0], coords2[0], 10)
        intp_val = np.polyval(coeff, xx_es)
        intpl_XX = np.hstack([intpl_XX, xx_es])
        intpl_YY = np.hstack([intpl_YY, intp_val])

    contour_smth = np.vstack([intpl_XX, intpl_YY]).T

    # find the crossing between vert_line and contours_rot.
    dist2 = distance.cdist(contour_smth, [top_atria_rot])
    min_dist2 = np.min(dist2)
    newy_atra = top_atria_rot[1] + min_dist2
    new_top_atria = [top_atria_rot[0], newy_atra]
    dist3 = distance.cdist(contour_smth, [new_top_atria])
    ind_min_dist3 = dist3.argmin()

    ind_alt_atria_top = contours_rot[:, 1].argmin()
    final_top_atria = np.asarray(
        [contours_rot[ind_alt_atria_top, 0], contours_rot[ind_alt_atria_top, 1]])
    final_perp_top_atria = contour_smth[ind_min_dist3, :]
    final_atrial_edge1 = np.asarray([xx_1, yy_1])
    final_atrial_edge2 = np.asarray([xx_2, yy_2])
    final_mid_avalve = np.asarray([xx_5, yy_5])

    if debug:
        plt.figure()
        plt.plot(contour_smth[:, 0], contour_smth[:, 1], 'r-')
        # plt.scatter(contour_smth[:, 0], contour_smth[:, 1])
        plt.plot(final_atrial_edge2[0], final_atrial_edge2[1], 'y*', label='atrium edge 2')
        plt.plot(final_atrial_edge1[0], final_atrial_edge1[1], 'm*', label='atrium edge 1')
        plt.plot(final_perp_top_atria[0], final_perp_top_atria[1], 'ko', label='perp top')
        plt.plot(final_top_atria[0], final_top_atria[1], 'c*', label='LA top')
        plt.plot(new_top_atria[0], new_top_atria[1], 'g*', label='new top')
        plt.plot(final_mid_avalve[0], final_mid_avalve[1], 'b*', label='midvalve')
        plt.plot([final_mid_avalve[0], final_top_atria[0]], [final_mid_avalve[1], final_top_atria[1]])
        plt.legend()
        plt.title('LA {}  frame {}'.format(_seq, _fr))

    # now find length of atrium divide in the  15 segments
    alength_top = distance.pdist([final_mid_avalve, final_top_atria])[0]
    alength_perp = distance.pdist([final_mid_avalve, final_perp_top_atria])[0]
    a_segmts = (final_mid_avalve[1] - final_top_atria[1]) / Nsegments_length

    a_diams = np.zeros(Nsegments_length)
    diam1 = abs(np.diff([xx_1, xx_2]))
    points_aux = np.zeros(((Nsegments_length - 1) * 2, 2))
    k = 0
    for ib in range(Nsegments_length):
        if ib == 0:
            a_diams[ib] = diam1
        else:
            vert_y = final_mid_avalve[1] - a_segmts * ib
            rgne_vertY = a_segmts / 6
            min_Y = vert_y - rgne_vertY
            max_Y = vert_y + rgne_vertY
            ind_sel_conts = np.where(np.logical_and(
                intpl_YY >= min_Y, intpl_YY <= max_Y))[0]

            if len(ind_sel_conts) == 0:
                logger.info('Frame {}, problem in disk {}'.format(_fr, ib))
                continue

            y_sel_conts = contour_smth[ind_sel_conts, 1]
            x_sel_conts = contour_smth[ind_sel_conts, 0]
            min_ys = np.argmin(np.abs(y_sel_conts - vert_y))

            p1 = ind_sel_conts[min_ys]
            point1 = contour_smth[p1]

            mean_x = np.mean([np.min(x_sel_conts), np.max(x_sel_conts)])
            if mean_x < point1[0]:
                ind_xs = np.where(contour_smth[ind_sel_conts, 0] < mean_x)[0]
                pts = contour_smth[ind_sel_conts[ind_xs], :]
                min_ys = np.argmin(np.abs(pts[:, 1] - vert_y))
                point2 = pts[min_ys]
                a_diam = distance.pdist([point1, point2])[0]

            elif np.min(x_sel_conts) == np.max(x_sel_conts):
                logger.info(
                    'Frame {}, disk {} diameter is zero'.format(_fr, ib))
                a_diam = 0
                point2 = np.zeros(2)
                point1 = np.zeros(2)
            else:
                ind_xs = np.where(contour_smth[ind_sel_conts, 0] > mean_x)[0]
                if len(ind_xs) > 0:
                    pts = contour_smth[ind_sel_conts[ind_xs], :]
                    min_ys = np.argmin(np.abs(pts[:, 1] - vert_y))
                    point2 = pts[min_ys]
                    a_diam = distance.pdist([point1, point2])[0]

                else:
                    a_diam = 0
                    point2 = np.zeros(2)
                    point1 = np.zeros(2)
                    logger.info(
                        'la_4Ch - Frame {}, disk {} diameter is zero'.format(_fr, ib))

            a_diams[ib] = a_diam
            points_aux[k, :] = point1
            points_aux[k + 1, :] = point2

            k += 2

    points_rotate = np.zeros(((Nsegments_length - 1) * 2 + 5, 2))
    points_rotate[0, :] = final_mid_avalve
    points_rotate[1, :] = final_top_atria
    points_rotate[2, :] = final_perp_top_atria
    points_rotate[3, :] = final_atrial_edge1
    points_rotate[4, :] = final_atrial_edge2
    points_rotate[5:, :] = points_aux

    radians2 = 2 * np.pi - radians
    points_non_rotate_ = np.zeros_like(points_rotate)
    for _jj, p in enumerate(points_non_rotate_):
        points_non_rotate_[_jj, 0] = points_rotate[_jj, 0] * math.cos(radians2) + points_rotate[_jj, 1] * math.sin(
            radians2)
        points_non_rotate_[_jj, 1] = -points_rotate[_jj, 0] * math.sin(radians2) + points_rotate[_jj, 1] * math.cos(
            radians2)
    if debug:
        plt.close('all')
    return a_diams, alength_top, alength_perp, points_non_rotate_, contours, lines_LV_, points_LV_


def detect_LV_points(seg, logger):
    myo_seg = np.squeeze(seg == 2).astype(float)
    kernel = np.ones((2, 2), np.uint8)
    # check if disconnected LV and use bloodpool to fill
    cnt_myo_seg = measure.find_contours(myo_seg, 0.8)
    if len(cnt_myo_seg) > 1:
        myo_seg = dilate_LV_myo(seg)
    myo2 = get_processed_myocardium(myo_seg, _label=1)
    cl_pts, _mid_valve = get_sorted_sk_pts(myo2, logger)
    dist_myo = distance.cdist(cl_pts, [_mid_valve])
    ind_apex = dist_myo.argmax()
    _apex = cl_pts[ind_apex, :]
    _septal_mv = cl_pts[0, 0], cl_pts[0, 1]
    _ant_mv = cl_pts[-1, 0], cl_pts[-1, 1]

    return np.asarray(_apex), np.asarray(_mid_valve), np.asarray(_septal_mv), np.asarray(_ant_mv)


def get_processed_myocardium(seg, _label=2):
    """
    This function tidies the LV myocardial segmentation, taking only the single
    largest connected component, and performing an opening (erosion+dilation)
    """

    myo_aux = np.squeeze(seg == _label).astype(float)  # get myocardial label
    myo_aux = closing(myo_aux, structure=np.ones((2, 2))).astype(float)
    cc_aux = measure.label(myo_aux, connectivity=1)
    ncc_aux = len(np.unique(cc_aux))

    if not ncc_aux <= 1:
        cc_counts, cc_inds = np.histogram(cc_aux, range(ncc_aux + 1))
        cc_inds = cc_inds[:-1]
        cc_inds_sorted = [_x for (y, _x) in sorted(zip(cc_counts, cc_inds))]
        # Take second largest CC (after background)
        biggest_cc_ind = cc_inds_sorted[-2]
        myo_aux = closing(myo_aux, structure=np.ones((2, 2))).astype(float)

        # Take largest connected component
        if not (len(np.where(cc_aux > 0)[0]) == len(np.where(cc_aux == biggest_cc_ind)[0])):
            mask = cc_aux == biggest_cc_ind
            myo_aux *= mask
            myo_aux = closing(myo_aux).astype(float)

    return myo_aux


def get_sorted_sk_pts(myo, logger, n_samples=48, centroid=np.array([0, 0])):
    #   ref -       reference start point for spline point ordering
    #   n_samples  output number of points for sampling spline

    # check for side branches? need connectivity check
    sk_im = skeletonize(myo)

    myo_pts = np.asarray(np.nonzero(myo)).transpose()
    sk_pts = np.asarray(np.nonzero(sk_im)).transpose()

    # convert to radial coordinates and sort circumferential
    if centroid[0] == 0 and centroid[1] == 0:
        centroid = np.mean(sk_pts, axis=0)

    # get skeleton consisting only of longest path
    sk_im = get_longest_path(sk_im, logger)

    # sort centreline points based from boundary points at valves as start
    # and end point. Make ref point out of LV through valve
    out = skeleton_endpoints(sk_im.astype(int))
    end_pts = np.asarray(np.nonzero(out)).transpose()
    sk_pts = np.asarray(np.nonzero(sk_im)).transpose()

    if len(end_pts) > 2:
        logger.info('Error! More than 2 end-points in LA myocardial skeleton.')
        cl_pts = []
        _mid_valve = []
        return cl_pts, _mid_valve
    else:
        # set reference to vector pointing from centroid to mid-valve
        _mid_valve = np.mean(end_pts, axis=0)
        ref = (_mid_valve - centroid) / norm(_mid_valve - centroid)
        sk_pts2 = sk_pts - centroid  # centre around centroid
        myo_pts2 = myo_pts - centroid
        theta = np.zeros([len(sk_pts2), ])
        theta_myo = np.zeros([len(myo_pts2), ])

        eps = 0.0001
        if len(sk_pts2) <= 5:
            logger.info(
                'Skeleton failed! Only of length {}'.format(len(sk_pts2)))
            cl_pts = []
            _mid_valve = []
            return cl_pts, _mid_valve
        else:
            # compute angle theta for skeleton points
            for k, ss in enumerate(sk_pts2):
                if (np.dot(ref, ss) / norm(ss) < 1.0 + eps) and (np.dot(ref, ss) / norm(ss) > 1.0 - eps):
                    theta[k] = 0
                elif (np.dot(ref, ss) / norm(ss) < -1.0 + eps) and (np.dot(ref, ss) / norm(ss) > -1.0 - eps):
                    theta[k] = 180
                else:
                    theta[k] = math.acos(
                        np.dot(ref, ss) / norm(ss)) * 180 / np.pi
                detp = ref[0] * ss[1] - ref[1] * ss[0]
                if detp > 0:
                    theta[k] = 360 - theta[k]
            thinds = theta.argsort()
            sk_pts = sk_pts[thinds, :].astype(
                float)  # ordered centreline points

            # # compute angle theta for myo points
            for k, ss in enumerate(myo_pts2):
                # compute angle theta
                eps = 0.0001
                if (np.dot(ref, ss) / norm(ss) < 1.0 + eps) and (np.dot(ref, ss) / norm(ss) > 1.0 - eps):
                    theta_myo[k] = 0
                elif (np.dot(ref, ss) / norm(ss) < -1.0 + eps) and (np.dot(ref, ss) / norm(ss) > -1.0 - eps):
                    theta_myo[k] = 180
                else:
                    theta_myo[k] = math.acos(
                        np.dot(ref, ss) / norm(ss)) * 180 / np.pi
                detp = ref[0] * ss[1] - ref[1] * ss[0]
                if detp > 0:
                    theta_myo[k] = 360 - theta_myo[k]
            # sub-sample and order myo points circumferential
            theta_myo.sort()

            # Remove duplicates
            sk_pts = binarymatrix(sk_pts)
            # fit b-spline curve to skeleton, sample fixed number of points
            tck, u = splprep(sk_pts.T, s=10.0, nest=-1, quiet=2)
            u_new = np.linspace(u.min(), u.max(), n_samples)
            cl_pts = np.zeros([n_samples, 2])
            cl_pts[:, 0], cl_pts[:, 1] = splev(u_new, tck)

            # get centreline theta
            cl_theta = np.zeros([len(cl_pts), ])
            cl_pts2 = cl_pts - centroid  # centre around centroid
            for k, ss in enumerate(cl_pts2):
                # compute angle theta
                if (np.dot(ref, ss) / norm(ss) < 1.0 + eps) and (np.dot(ref, ss) / norm(ss) > 1.0 - eps):
                    cl_theta[k] = 0
                else:
                    cl_theta[k] = math.acos(
                        np.dot(ref, ss) / norm(ss)) * 180 / np.pi
                detp = ref[0] * ss[1] - ref[1] * ss[0]
                if detp > 0:
                    cl_theta[k] = 360 - cl_theta[k]
            cl_theta.sort()
            return cl_pts, _mid_valve


def get_longest_path(skel, logger):
    # first create edges from skeleton
    sk_im = skel.copy()
    # remove bad (L-shaped) junctions
    sk_im = remove_bad_junctions(sk_im, logger)

    # get seeds for longest path from existing end-points
    out = skeleton_endpoints(sk_im.astype(int))
    end_pts = np.asarray(np.nonzero(out)).transpose()
    if len(end_pts) == 0:
        logger.info('ERROR! No end-points detected! Exiting.')
    # break
    elif len(end_pts) == 1:
        logger.info('Warning! Only 1 end-point detected!')
    elif len(end_pts) > 2:
        logger.info('Warning! {} end-points detected!'.format(len(end_pts)))

    sk_pts = np.asarray(np.nonzero(sk_im)).transpose()
    # search indices of sk_pts for end points
    tmp_inds = np.ravel_multi_index(
        sk_pts.T, (np.max(sk_pts[:, 0]) + 1, np.max(sk_pts[:, 1]) + 1))
    seed_inds = np.zeros((len(end_pts), 1))
    for i, e in enumerate(end_pts):
        seed_inds[i] = int(
            np.where(tmp_inds == np.ravel_multi_index(e.T, (np.max(sk_pts[:, 0]) + 1, np.max(sk_pts[:, 1]) + 1)))[0])
    sk_im_inds = np.zeros_like(sk_im, dtype=int)

    for i, p in enumerate(sk_pts):
        sk_im_inds[p[0], p[1]] = i

    kernel1 = np.uint8([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])
    edges = []
    for i, p in enumerate(sk_pts):
        mask = sk_im_inds[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2]
        o = np.multiply(kernel1, mask)
        for c in o[o > 0]:
            edges.append(['{}'.format(i), '{}'.format(c)])
    # create graph
    G = defaultdict(list)
    for (ss, t) in edges:
        if t not in G[ss]:
            G[ss].append(t)
        if ss not in G[t]:
            G[t].append(ss)
    # find max path
    max_path = []
    for j in range(len(seed_inds)):
        all_paths = depth_first_search(G, str(int(seed_inds[j][0])))
        max_path2 = max(all_paths, key=lambda l: len(l))
        if len(max_path2) > len(max_path):
            max_path = max_path2
    # create new image only with max path
    sk_im_maxp = np.zeros_like(sk_im, dtype=int)
    for j in max_path:
        p = sk_pts[int(j)]
        sk_im_maxp[p[0], p[1]] = 1
    return sk_im_maxp


def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    # now look through to find the value of 11
    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 1

    return out


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def detect_RV_points(_seg, septal_mv, logger):
    rv_seg = np.squeeze(_seg == 3).astype(float)

    sk_pts = measure.find_contours(rv_seg, 0.8)
    if len(sk_pts) > 1:
        nb_pts = []
        for ll in range(len(sk_pts)):
            nb_pts.append(len(sk_pts[ll]))
        sk_pts = sk_pts[np.argmax(nb_pts)]
    sk_pts = np.squeeze(sk_pts)
    sk_pts = np.unique(sk_pts, axis=0)
    centroid = np.mean(sk_pts, axis=0)

    _lv_valve = closest_node(np.squeeze(septal_mv), sk_pts)
    ref = (_lv_valve - centroid) / norm(_lv_valve - centroid)

    sk_pts2 = sk_pts - centroid  # centre around centroid
    theta = np.zeros([len(sk_pts2), ])

    eps = 0.0001
    if len(sk_pts2) <= 5:
        logger.info('Skeleton failed! Only of length {}'.format(len(sk_pts2)))
        _cl_pts = []
    else:
        # compute angle theta for skeleton points
        for k, ss in enumerate(sk_pts2):
            if (np.dot(ref, ss) / norm(ss) < 1.0 + eps) and (np.dot(ref, ss) / norm(ss) > 1.0 - eps):
                theta[k] = 0
            elif (np.dot(ref, ss) / norm(ss) < -1.0 + eps) and (np.dot(ref, ss) / norm(ss) > -1.0 - eps):
                theta[k] = 180
            else:
                theta[k] = math.acos(np.dot(ref, ss) / norm(ss)) * 180 / np.pi
            detp = ref[0] * ss[1] - ref[1] * ss[0]
            if detp > 0:
                theta[k] = 360 - theta[k]
        thinds = theta.argsort()
        sk_pts = sk_pts[thinds, :].astype(float)  # ordered centreline points

        # Remove duplicates
        sk_pts = binarymatrix(sk_pts)
        # fit b-spline curve to skeleton, sample fixed number of points
        tck, u = splprep(sk_pts.T, s=10.0, per=1, quiet=2)

        num_points = sk_pts.shape[0]  # 80
        u_new = np.linspace(u.min(), u.max(), num_points)
        _cl_pts = np.zeros([num_points, 2])
        _cl_pts[:, 0], _cl_pts[:, 1] = splev(u_new, tck)

    dist_rv = distance.cdist(_cl_pts, [_lv_valve])
    _ind_apex = dist_rv.argmax()
    _apex_RV = _cl_pts[_ind_apex, :]

    m = np.diff(_cl_pts[:, 0]) / np.diff(_cl_pts[:, 1])
    angle = np.arctan(m) * 180 / np.pi
    idx = np.sign(angle)
    _ind_free_wall = np.where(idx == -1)[0]

    _area = 10000 * np.ones(len(_ind_free_wall))
    for ai, ind in enumerate(_ind_free_wall):
        AB = np.linalg.norm(_lv_valve - _apex_RV)
        BC = np.linalg.norm(_lv_valve - _cl_pts[ind, :])
        AC = np.linalg.norm(_cl_pts[ind, :] - _apex_RV)
        if AC > 10 and BC > 10:
            _area[ai] = np.abs(AB ** 2 + BC ** 2 - AC ** 2)
    _free_rv_point = _cl_pts[_ind_free_wall[_area.argmin()], :]

    if debug:
        plt.figure()
        plt.plot(_cl_pts[:, 1], _cl_pts[:, 0])
        plt.plot(_free_rv_point[1], _free_rv_point[0], 'r*', label='free rv')
        plt.plot(_apex_RV[1], _apex_RV[0], 'b*', label='apex rv')
        plt.plot(_lv_valve[1], _lv_valve[0], 'k*', label='rvlv')
        plt.legend()
        plt.title('RV')
        plt.close('all')

    return np.asarray(_apex_RV), np.asarray(_lv_valve), np.asarray(_free_rv_point)


def remove_bad_junctions(skel, logger):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # kernel_A used for unnecessary nodes in L-shaped junctions (retain diags)
    kernels_A = [np.uint8([[0, 1, 0],
                           [1, 10, 1],
                           [0, 1, 0]])]
    src_depth = -1
    for k in kernels_A:
        filtered = cv2.filter2D(skel, src_depth, k)
        skel[filtered >= 13] = 0
        if len(np.where(filtered == 14)[0]) > 0:
            logger.info('Warning! You have a 3x3 loop!')

    return skel


def depth_first_search(G, v, seen=None, path=None):
    if seen is None:
        seen = []
    if path is None:
        path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(depth_first_search(G, t, seen, t_path))
    return paths


def calculate_tt(seg_file):
    nim = nib.load(seg_file)
    nim_array = nim.get_fdata()
    nim_hdr = nim.header
    pix_dim = nim_hdr["pixdim"][4]
    tt = pix_dim * np.array(range(0, nim_array.shape[3]))
    return tt


def compute_atria_params(study_ID, subject_dir, results_dir, logger, recalculate_points=False):
    window_size, poly_order = 9, 3
    QC_atria_2Ch, QC_atria_4Ch_LA, QC_atria_4Ch_RA = 0, 0, 0
    s = 300  # Spline smoothing parameter for the strain curves
    p = 2  # Multiplier for increasing number of points in strain curve
    save_2ch_dict_flag = False  # Whether to save 2ch dict and points

    # =========================================================================
    # la_2Ch - calculate area and points
    # =========================================================================
    filename_la_seg_2Ch = os.path.join(subject_dir, 'la_2Ch_seg_nnUnet.nii.gz')
    if os.path.exists(filename_la_seg_2Ch):
        nim = nib.load(filename_la_seg_2Ch)
        la_seg_2Ch = nim.get_fdata()
        dx, dy, _ = nim.header['pixdim'][1:4]
        area_per_voxel = dx * dy
        if len(la_seg_2Ch.shape) == 4:
            la_seg_2Ch = la_seg_2Ch[:, :, 0, :]
        _, _, N_frames_2Ch = la_seg_2Ch.shape
        tt_2ch_orig = calculate_tt(filename_la_seg_2Ch)

        # Get largest connected components
        for fr in range(N_frames_2Ch):
            la_seg_2Ch[:, :, fr] = getLargestCC(la_seg_2Ch[:, :, fr])

        # Compute 2ch area using number of pixels
        area_LA_2Ch = -1*np.ones(N_frames_2Ch)
        for fr in range(N_frames_2Ch):
            area_LA_2Ch[fr] = np.sum(
                np.squeeze(
                    la_seg_2Ch[:, :, fr] == 3).astype(float)) * area_per_voxel

        # Compute 2ch params needed for simpson's rule
        dict_2ch_file = os.path.join(  # dict of length values
            results_dir, f'{study_ID}_2ch_length_dict.npy')
        if os.path.exists(dict_2ch_file) and not recalculate_points:
            # If already saved, load the dictionary
            logger.info('Loading pre-saved dictionary of 2ch params')
            dict_2ch = np.load(dict_2ch_file, allow_pickle=True).item()
            la_diams_2Ch = dict_2ch['la_diams_2Ch']
            length_top_2Ch = dict_2ch['length_top_2Ch']
            LA_circumf_cycle_2Ch = dict_2ch['LA_circumf_cycle_2Ch']
        else:
            # Otherwise, calculate and save points
            logger.info('Calculating 2ch points')
            save_2ch_dict_flag = True
            points_LV_2Ch = -1*np.ones((N_frames_2Ch, 4, 2))
            LV_atria_points_2Ch = -1*np.ones((N_frames_2Ch, 9, 2))
            la_diams_2Ch = -1*np.ones((N_frames_2Ch, Nsegments_length))
            length_top_2Ch = -1*np.ones(N_frames_2Ch)
            LA_circumf_cycle_2Ch = -1*np.ones(N_frames_2Ch)

            for fr in range(N_frames_2Ch):
                try:
                    apex, mid_valve, anterior, inferior = detect_LV_points(
                        la_seg_2Ch[:, :, fr], logger)
                    points = np.vstack([apex, mid_valve, anterior, inferior])
                    points_LV_2Ch[fr, :] = points
                except Exception:
                    logger.error(
                        f'Problem detecting LV points {study_ID}'
                        f' in la_2Ch fr {fr}')
                    QC_atria_2Ch = 1

                if QC_atria_2Ch == 0:
                    try:
                        la_dia, lentop, lenperp, points_non_rotate, contours_LA, lines_LV, points_LV = \
                            get_left_atrial_volumes(
                                la_seg_2Ch[:, :, fr], 'la_2Ch', fr, points, logger)
                        la_diams_2Ch[fr, :] = la_dia * dx
                        length_top_2Ch[fr] = lentop * dx
                        # final_mid_avalve
                        LV_atria_points_2Ch[fr, 0, :] = points_non_rotate[0, :]
                        # final_top_atria
                        LV_atria_points_2Ch[fr, 1, :] = points_non_rotate[1, :]
                        # final_perp_top_atria
                        LV_atria_points_2Ch[fr, 2, :] = points_non_rotate[2, :]
                        # final_atrial_edge1
                        LV_atria_points_2Ch[fr, 3, :] = points_non_rotate[3, :]
                        # final_atrial_edge2
                        LV_atria_points_2Ch[fr, 4, :] = points_non_rotate[4, :]
                        LV_atria_points_2Ch[fr, 5, :] = points_LV[0, :]  # apex
                        # mid_valve
                        LV_atria_points_2Ch[fr, 6, :] = points_LV[1, :]
                        # inferior_2Ch
                        LV_atria_points_2Ch[fr, 7, :] = points_LV[2, :]
                        # anterior_2Ch
                        LV_atria_points_2Ch[fr, 8, :] = points_LV[3, :]

                        # compute atrial circumference
                        LA_circumf_2Ch = get_atrial_circumference(contours_LA, [points_non_rotate[3, 1], points_non_rotate[3, 0]], [
                            points_non_rotate[4, 1], points_non_rotate[4, 0]], [points_non_rotate[1, 1], points_non_rotate[1, 0]], fr, study_ID)
                        LA_circumf_len_2Ch = LA_circumf_2Ch * dx
                        LA_circumf_cycle_2Ch[fr] = LA_circumf_len_2Ch
                    except Exception:
                        logger.error('Problem in disk-making with subject {} in la_2Ch fr {}'.
                                     format(study_ID, fr))
                        QC_atria_2Ch = 1
            logger.info('Finished calculating 2ch points\n')

        # =====================================================================
        # Compute 2ch strain
        # =====================================================================
        if QC_atria_2Ch == 0:
            try:
                # Strain
                LA_strain_circum_2Ch = get_strain(LA_circumf_cycle_2Ch)
                LA_strain_longitud_2ch = get_strain(length_top_2Ch)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strain_circum_2Ch.txt'), LA_strain_circum_2Ch)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strain_longitud_2Ch.txt'), LA_strain_longitud_2ch)

                # x = np.linspace(0, N_frames_2Ch - 1, N_frames_2Ch)
                # xx = np.linspace(np.min(x), np.max(x), N_frames_2Ch)
                # itp = interp1d(x, LA_strain_circum_2Ch)
                # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                # LA_strain_circum_2Ch_smooth = yy_sg
                # itp = interp1d(x, LA_strain_longitud_2ch)
                # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                # LA_strain_longitud_2ch_smooth = yy_sg
                LA_strain_circum_2Ch_smooth, tt_2ch = smooth_curve_1d(LA_strain_circum_2Ch, N_frames_2Ch, s, N_frames_2Ch*p, tt_2ch_orig)
                LA_strain_longitud_2ch_smooth, tt_2ch = smooth_curve_1d(LA_strain_longitud_2ch, N_frames_2Ch, s, N_frames_2Ch*p, tt_2ch_orig)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strain_circum_2Ch_smooth.txt'), LA_strain_circum_2Ch_smooth)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strain_longitud_2ch_smooth.txt'), LA_strain_longitud_2ch_smooth)

                # Strain rate
                LA_strainRate_circum_2Ch = np.gradient(LA_strain_circum_2Ch_smooth, tt_2ch)
                x = np.linspace(0, N_frames_2Ch*p - 1, N_frames_2Ch*p)
                xx = np.linspace(np.min(x), np.max(x), N_frames_2Ch*p)
                itp = interp1d(x, LA_strainRate_circum_2Ch)
                LA_strainRate_circum_2Ch_smooth = savgol_filter(itp(xx), window_size, poly_order)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strainRate_circum_2Ch.txt'), LA_strainRate_circum_2Ch)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strainRate_circum_2Ch_smooth.txt'), LA_strainRate_circum_2Ch_smooth)

                LA_strainRate_longitud_2ch = np.gradient(LA_strain_longitud_2ch_smooth, tt_2ch)
                itp = interp1d(x, LA_strainRate_longitud_2ch)
                LA_strainRate_longitud_2ch_smooth = savgol_filter(itp(xx), window_size, poly_order)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strainRate_longitud_2Ch.txt'), LA_strainRate_longitud_2ch)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strainRate_longitud_2Ch_smooth.txt'), LA_strainRate_longitud_2ch_smooth)
            except:
                logger.error('Problem calculating 2ch strain and strain rate')
                LA_strain_circum_2Ch = -1*np.ones(N_frames_2Ch)
                LA_strain_longitud_2ch = -1*np.ones(N_frames_2Ch)
                LA_strainRate_circum_2Ch = -1*np.ones(N_frames_2Ch)
                LA_strainRate_longitud_2ch = -1*np.ones(N_frames_2Ch)
                QC_atria_2Ch = 1
    else:
        logger.info('No 2ch image - skipping.')
        return [None, None]

    # Save 2ch dict and atrial points to avoid recomupting
    if save_2ch_dict_flag and QC_atria_2Ch == 0:
        dict_2ch = {}
        dict_2ch['la_diams_2Ch'] = la_diams_2Ch
        dict_2ch['length_top_2Ch'] = length_top_2Ch
        dict_2ch['LA_circumf_cycle_2Ch'] = LA_circumf_cycle_2Ch

        np.save(dict_2ch_file, dict_2ch)
        np.save(os.path.join(
            results_dir, f'{study_ID}_LV_atria_points_2Ch.npy'),
            LV_atria_points_2Ch)
        np.save(os.path.join(results_dir, 'points_LV_2Ch.npy'), points_LV_2Ch)

    # =====================================================================
    # Compute 2ch volume
    # =====================================================================
    if QC_atria_2Ch == 0:
        LA_volume_area_2ch = -1*np.ones(N_frames_2Ch)
        LA_volume_SR_2ch = -1*np.ones(N_frames_2Ch)
        for fr in range(N_frames_2Ch):
            # Simpson's rule
            d1d2 = la_diams_2Ch[fr, :] * la_diams_2Ch[fr, :]
            length = np.min([length_top_2Ch[fr], length_top_2Ch[fr]])
            LA_volume_SR_2ch[fr] = math.pi / 4 * length * \
                np.sum(d1d2) / Nsegments_length / 1000

            # Area method
            LA_volume_area_2ch[fr] = 0.85 * area_LA_2Ch[fr] * \
                area_LA_2Ch[fr] / length / 1000

        x = np.linspace(0, N_frames_2Ch - 1, N_frames_2Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_2Ch)
        itp = interp1d(x, LA_volume_SR_2ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volume_SR_2ch_smooth = yy_sg
        itp = interp1d(x, LA_volume_area_2ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volume_area_2ch_smooth = yy_sg

        np.savetxt(os.path.join(
            results_dir, 'LA_volume_SR_2ch.txt'), LA_volume_SR_2ch)
        np.savetxt(os.path.join(
            results_dir, 'LA_volume_SR_2ch_smooth.txt'),
            LA_volume_SR_2ch_smooth)
        np.savetxt(os.path.join(
            results_dir, 'LA_volume_area_2ch.txt'), LA_volume_area_2ch)
        np.savetxt(os.path.join(
            results_dir, 'LA_volume_area_2ch_smooth.txt'),
            LA_volume_area_2ch_smooth)
    else:
        LA_volume_SR_2ch_smooth = -1*np.ones(N_frames_2Ch)
        LA_volume_area_2ch_smooth = -1*np.ones(N_frames_2Ch)

    # =========================================================================
    # la_4Ch - calculate area and points
    # =========================================================================
    filename_la_seg_4Ch = os.path.join(subject_dir, 'la_4Ch_seg_nnUnet.nii.gz')
    save_4ch_LA_dict_flag = False  # Whether to save 4ch dict and points
    save_4ch_RA_dict_flag = False
    if os.path.exists(filename_la_seg_4Ch):
        nim = nib.load(filename_la_seg_4Ch)
        la_seg_4Ch = nim.get_fdata()
        dx, dy, _ = nim.header['pixdim'][1:4]
        area_per_voxel = dx * dy
        if len(la_seg_4Ch.shape) == 4:
            la_seg_4Ch = la_seg_4Ch[:, :, 0, :]
        la_seg_4Ch = np.transpose(la_seg_4Ch, [1, 0, 2])
        _, _, N_frames_4Ch = la_seg_4Ch.shape
        tt_4ch_orig = calculate_tt(filename_la_seg_4Ch)

        # Get largest connected components
        for fr in range(N_frames_4Ch):
            la_seg_4Ch[:, :, fr] = getLargestCC(la_seg_4Ch[:, :, fr])

        # Compute 4ch area using number of pixels
        area_LA_4Ch = -1*np.ones(N_frames_4Ch)
        area_RA = -1*np.ones(N_frames_4Ch)  # LA_4Ch
        for fr in range(N_frames_4Ch):
            area_LA_4Ch[fr] = np.sum(
                np.squeeze(
                    la_seg_4Ch[:, :, fr] == 4).astype(float)) * area_per_voxel
            area_RA[fr] = np.sum(np.squeeze(la_seg_4Ch[:, :, fr] == 5).astype(
                float)) * area_per_voxel  # in mm2

        # Compute 4ch params needed for simpson's rule
        dict_4ch_LA_file = os.path.join(  # dict of length values - LA
            results_dir, f'{study_ID}_4ch_LA_length_dict.npy')
        dict_4ch_RA_file = os.path.join(  # dict of length values - RA
            results_dir, f'{study_ID}_4ch_RA_length_dict.npy')
        points_LV_4ch_file = os.path.join(results_dir, 'points_LV_4Ch.npy') 
        if os.path.exists(dict_4ch_LA_file) and not recalculate_points:
            # If already saved, load the dictionary
            logger.info('Loading pre-saved dictionary of 4ch LA params')
            dict_4ch_LA = np.load(dict_4ch_LA_file, allow_pickle=True).item()
            la_diams_4Ch = dict_4ch_LA['la_diams_4Ch']
            length_top_4Ch = dict_4ch_LA['length_top_4Ch']
            LA_circumf_cycle_4Ch = dict_4ch_LA['LA_circumf_cycle_4Ch']
            points_LV_4Ch = np.load(points_LV_4ch_file)
        else:
            # Otherwise calculate and save points
            save_4ch_LA_dict_flag = True
            la_diams_4Ch = -1*np.ones((N_frames_4Ch, Nsegments_length))
            length_top_4Ch = -1*np.ones(N_frames_4Ch)
            LA_circumf_cycle_4Ch = -1*np.ones(N_frames_4Ch)
            points_LV_4Ch = -1*np.ones((N_frames_4Ch, 4, 2))
            LV_atria_points_4Ch = -1*np.ones((N_frames_4Ch, 9, 2))
            logger.info('Calculating 4ch LA points')

            for fr in range(N_frames_4Ch):
                try:
                    apex, mid_valve, anterior, inferior = detect_LV_points(
                        la_seg_4Ch[:, :, fr], logger)
                    points = np.vstack([apex, mid_valve, anterior, inferior])
                    points_LV_4Ch[fr, :] = points
                except Exception:
                    logger.error(
                        f'Problem detecting LV points {study_ID}'
                        f' in la_4Ch fr {fr}')
                    QC_atria_4Ch_LA = 1

                if QC_atria_4Ch_LA == 0:
                    # LA/LV points
                    try:
                        la_dia, lentop, _, points_non_rotate, contours_LA, _, points_LV = \
                            get_left_atrial_volumes(
                                la_seg_4Ch[:, :, fr], 'la_4Ch', fr, points, logger)
                        la_diams_4Ch[fr, :] = la_dia * dx
                        length_top_4Ch[fr] = lentop * dx
                        # final_mid_avalve
                        LV_atria_points_4Ch[fr, 0, :] = points_non_rotate[0, :]
                        # final_top_atria
                        LV_atria_points_4Ch[fr, 1, :] = points_non_rotate[1, :]
                        # final_perp_top_atria
                        LV_atria_points_4Ch[fr, 2, :] = points_non_rotate[2, :]
                        # final_atrial_edge1
                        LV_atria_points_4Ch[fr, 3, :] = points_non_rotate[3, :]
                        # final_atrial_edge2
                        LV_atria_points_4Ch[fr, 4, :] = points_non_rotate[4, :]
                        LV_atria_points_4Ch[fr, 5, :] = points_LV[0, :]  # apex
                        # mid_valve
                        LV_atria_points_4Ch[fr, 6, :] = points_LV[1, :]
                        # lateral_4Ch
                        LV_atria_points_4Ch[fr, 7, :] = points_LV[2, :]
                        # septal_4Ch
                        LV_atria_points_4Ch[fr, 8, :] = points_LV[3, :]
                        # compute atrial circumference
                        LA_circumf_4Ch = get_atrial_circumference(contours_LA, [points_non_rotate[3, 1], points_non_rotate[3, 0]], [
                            points_non_rotate[4, 1], points_non_rotate[4, 0]], [points_non_rotate[1, 1], points_non_rotate[1, 0]], fr, study_ID)
                        LA_circumf_len_4Ch = LA_circumf_4Ch * dx
                        LA_circumf_cycle_4Ch[fr] = LA_circumf_len_4Ch
                    except Exception:
                        logger.error('Problem in disk-making with subject {} in la_4Ch fr {}'.
                                     format(study_ID, fr))
                        return [None, None]
            logger.info('Finished calculating 4ch LA points\n')

        if os.path.exists(dict_4ch_RA_file) and not recalculate_points:
            logger.info('Loading pre-saved dictionary of 4ch RA params')
            dict_4ch_RA = np.load(dict_4ch_RA_file, allow_pickle=True).item()
            la_diams_RV = dict_4ch_RA['la_diams_RV']
            length_top_RV = dict_4ch_RA['length_top_RV']
            RA_circumf_cycle_4Ch = dict_4ch_RA['RA_circumf_cycle_4Ch']
        else:
            # Otherwise calculate and save points
            save_4ch_RA_dict_flag = True
            la_diams_RV = -1*np.ones((N_frames_4Ch, Nsegments_length))  # LA_4Ch
            length_top_RV = -1*np.ones(N_frames_4Ch)  # LA_4Ch
            RA_circumf_cycle_4Ch = -1*np.ones(N_frames_4Ch)
            points_RV_4Ch = -1*np.ones((N_frames_4Ch, 3, 2))
            RV_atria_points_4Ch = -1*np.ones((N_frames_4Ch, 8, 2))
            logger.info('Calculating 4ch RA points')

            for fr in range(N_frames_4Ch):
                try:
                    # seg, anterior, log
                    apex_RV, rvlv_point, free_rv_point = detect_RV_points(
                        la_seg_4Ch[:, :, fr], points_LV_4Ch[fr, 2], logger)
                    pointsRV = np.vstack([apex_RV, rvlv_point, free_rv_point])
                    points_RV_4Ch[fr, :] = pointsRV
                except Exception:
                    logger.error(
                        f'Problem detecting RV points {study_ID}'
                        f' in la_4Ch fr {fr}')
                    QC_atria_4Ch_RA = 1

                if QC_atria_4Ch_RA == 0:
                    # RA/RV points
                    try:
                        la_dia, lentop, _, points_non_rotate, contours_RA, _ = \
                            get_right_atrial_volumes(
                                la_seg_4Ch[:, :, fr], fr, pointsRV, logger)

                        la_diams_RV[fr, :] = la_dia * dx
                        length_top_RV[fr] = lentop * dx

                        # final_mid_avalve
                        RV_atria_points_4Ch[fr, 0, :] = points_non_rotate[0, :]
                        # final_top_atria
                        RV_atria_points_4Ch[fr, 1, :] = points_non_rotate[1, :]
                        # final_perp_top_atria
                        RV_atria_points_4Ch[fr, 2, :] = points_non_rotate[2, :]
                        # final_atrial_edge1
                        RV_atria_points_4Ch[fr, 3, :] = points_non_rotate[3, :]
                        # final_atrial_edge2
                        RV_atria_points_4Ch[fr, 4, :] = points_non_rotate[4, :]
                        RV_atria_points_4Ch[fr, 5,
                                            :] = pointsRV[0, :]  # apex_RV
                        # rvlv_point
                        RV_atria_points_4Ch[fr, 6, :] = pointsRV[1, :]
                        # free_rv_point
                        RV_atria_points_4Ch[fr, 7, :] = pointsRV[2, :]
                        # compute atrial circumference
                        RA_circumf_4Ch = get_atrial_circumference(contours_RA, [points_non_rotate[3, 1], points_non_rotate[3, 0]], [
                            points_non_rotate[4, 1], points_non_rotate[4, 0]], [points_non_rotate[1, 1], points_non_rotate[1, 0]], fr, study_ID)
                        RA_circumf_len_4Ch = RA_circumf_4Ch * dx
                        RA_circumf_cycle_4Ch[fr] = RA_circumf_len_4Ch
                    except Exception:
                        logger.error(
                            'RV Problem in disk-making with subject {} in la_4Ch fr {}'.format(study_ID, fr))
                        QC_atria_4Ch_RA = 1
            logger.info('Finished calculating 4ch RA points\n')

        # Save 4ch atrial points to avoid recomupting
        if save_4ch_LA_dict_flag and QC_atria_4Ch_LA == 0:
            dict_4ch_LA = {}
            dict_4ch_LA['la_diams_4Ch'] = la_diams_4Ch
            dict_4ch_LA['length_top_4Ch'] = length_top_4Ch
            dict_4ch_LA['LA_circumf_cycle_4Ch'] = LA_circumf_cycle_4Ch

            np.save(dict_4ch_LA_file, dict_4ch_LA)
            np.save(os.path.join(results_dir, f'{study_ID}_LV_atria_points_4Ch.npy'), LV_atria_points_4Ch)
            np.save(points_LV_4ch_file, points_LV_4Ch)

        if save_4ch_RA_dict_flag and QC_atria_4Ch_RA == 0:
            dict_4ch_RA = {}
            dict_4ch_RA['la_diams_RV'] = la_diams_RV
            dict_4ch_RA['length_top_RV'] = length_top_RV
            dict_4ch_RA['RA_circumf_cycle_4Ch'] = RA_circumf_cycle_4Ch
            np.save(dict_4ch_RA_file, dict_4ch_RA)
            np.save(os.path.join(results_dir, f'{study_ID}_RV_atria_points_4Ch.npy'), RV_atria_points_4Ch)
            np.save(os.path.join(results_dir, 'points_RV_4Ch.npy'), points_RV_4Ch)

        # =====================================================================
        # 4ch Strain
        # =====================================================================
        if QC_atria_4Ch_LA == 0:
            LA_strain_circum_4Ch_smooth = -1*np.ones(N_frames_4Ch)
            LA_strain_longitud_4Ch_smooth = -1*np.ones(N_frames_4Ch)
            LA_strainRate_circum_4Ch_smooth = -1*np.ones(N_frames_4Ch)
            LA_strainRate_longitud_4Ch_smooth = -1*np.ones(N_frames_4Ch)
            try:
                # Strain
                LA_strain_circum_4Ch = get_strain(LA_circumf_cycle_4Ch)
                x = np.linspace(0, N_frames_4Ch*p - 1, N_frames_4Ch*p)
                xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch*p)
                # itp = interp1d(x, LA_strain_circum_4Ch)
                # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                # LA_strain_circum_4Ch_smooth = yy_sg
                LA_strain_circum_4Ch_smooth, tt_4ch = smooth_curve_1d(LA_strain_circum_4Ch, N_frames_4Ch, s, N_frames_4Ch*p, tt_4ch_orig)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strain_circum_4Ch_smooth.txt'),
                    LA_strain_circum_4Ch_smooth)

                LA_strain_longitud_4Ch = get_strain(length_top_4Ch)
                LA_strain_longitud_4Ch_smooth, tt_4ch = smooth_curve_1d(LA_strain_longitud_4Ch, N_frames_4Ch, s, N_frames_4Ch*p, tt_4ch_orig)
                # itp = interp1d(x, LA_strain_longitud_4Ch)
                # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                # LA_strain_longitud_4Ch_smooth = yy_sg
                np.savetxt(os.path.join(
                    results_dir, 'LA_strain_longitud_4Ch_smooth.txt'),
                    LA_strain_longitud_4Ch_smooth)
                
                # Strain rate
                LA_strainRate_circum_4Ch = np.gradient(LA_strain_circum_4Ch_smooth, tt_4ch)
                itp = interp1d(x, LA_strainRate_circum_4Ch)
                LA_strainRate_circum_4Ch_smooth = savgol_filter(itp(xx), window_size, poly_order)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strainRate_circum_4Ch_smooth.txt'),
                    LA_strainRate_circum_4Ch_smooth)
                
                LA_strainRate_longitud_4Ch = np.gradient(LA_strain_longitud_4Ch_smooth, tt_4ch)
                itp = interp1d(x, LA_strainRate_longitud_4Ch)
                LA_strainRate_longitud_4Ch_smooth = savgol_filter(itp(xx), window_size, poly_order)
                np.savetxt(os.path.join(
                    results_dir, 'LA_strainRate_longitud_4Ch_smooth.txt'),
                    LA_strainRate_longitud_4Ch_smooth)
            except:
                QC_atria_4Ch_LA = 1
                logger.error('Error interpolating LA strains')

        RA_strain_circum_4Ch_smooth = -1*np.ones(50)
        RA_strain_longitud_4Ch_smooth = -1*np.ones(50)
        RA_strainRate_circum_4Ch_smooth = -1*np.ones(50)
        RA_strainRate_longitud_4Ch_smooth = -1*np.ones(50)
        if QC_atria_4Ch_RA == 0:
            try:
                # Strain
                RA_strain_circum_4Ch = get_strain(RA_circumf_cycle_4Ch)
                RA_strain_circum_4Ch_smooth, tt_4ch = smooth_curve_1d(RA_strain_circum_4Ch, N_frames_4Ch, s, N_frames_4Ch*p, tt_4ch_orig)
                x = np.linspace(0, N_frames_4Ch*p - 1, N_frames_4Ch*p)
                xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch*p)
                # itp = interp1d(x, RA_strain_circum_4Ch)
                # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                # RA_strain_circum_4Ch_smooth = yy_sg
                np.savetxt(os.path.join(
                    results_dir, 'RA_strain_circum_4Ch_smooth.txt'),
                    RA_strain_circum_4Ch_smooth)

                RA_strain_longitud_4Ch = get_strain(length_top_RV)
                RA_strain_longitud_4Ch_smooth, tt_4ch = smooth_curve_1d(RA_strain_longitud_4Ch, N_frames_4Ch, s, N_frames_4Ch*p, tt_4ch_orig)
                # itp = interp1d(x, RA_strain_longitud_4Ch)
                # yy_sg = savgol_filter(itp(xx), window_size, poly_order)
                # RA_strain_longitud_4Ch_smooth = yy_sg
                np.savetxt(os.path.join(
                    results_dir, 'RA_strain_longitud_4Ch_smooth.txt'),
                    RA_strain_longitud_4Ch_smooth)

                # Strain rate
                RA_strainRate_circum_4Ch = np.gradient(RA_strain_circum_4Ch_smooth, tt_4ch)
                itp = interp1d(x, RA_strainRate_circum_4Ch)
                RA_strainRate_circum_4Ch_smooth = savgol_filter(itp(xx), window_size, poly_order)
                np.savetxt(os.path.join(
                    results_dir, 'RA_strainRate_circum_4Ch_smooth.txt'),
                    RA_strainRate_circum_4Ch_smooth)
                
                RA_strainRate_longitud_4Ch = np.gradient(RA_strain_longitud_4Ch_smooth, tt_4ch)
                itp = interp1d(x, RA_strainRate_longitud_4Ch)
                RA_strainRate_longitud_4Ch_smooth = savgol_filter(itp(xx), window_size, poly_order)
                np.savetxt(os.path.join(
                    results_dir, 'RA_strainRate_longitud_4Ch_smooth.txt'),
                    RA_strainRate_longitud_4Ch_smooth)
            except:
                QC_atria_4Ch_RA = 1
                logger.error('Error interpolating RA strains')
    else:
        logger.info('No 4ch image - skipping.')
        return [None, None]
    
    # =====================================================================
    # Compute 4ch volume
    # =====================================================================
    # LA volumes
    if QC_atria_4Ch_LA == 0:
        LA_volume_SR_4Ch = -1*np.ones(N_frames_4Ch)
        LA_volume_area_4ch = -1*np.ones(N_frames_4Ch)
        for fr in range(N_frames_4Ch):
            # Simpson's rule
            d1d2 = la_diams_4Ch[fr, :] * la_diams_4Ch[fr, :]
            length = np.min([length_top_4Ch[fr], length_top_4Ch[fr]])
            LA_volume_SR_4Ch[fr] = math.pi / 4 * length * \
                np.sum(d1d2) / Nsegments_length / 1000

            # Area method
            LA_volume_area_4ch[fr] = 0.85 * area_LA_4Ch[fr] * \
                area_LA_4Ch[fr] / length / 1000

        x = np.linspace(0, N_frames_4Ch - 1, N_frames_4Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch)
        itp = interp1d(x, LA_volume_SR_4Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volume_SR_4Ch_smooth = yy_sg
        itp = interp1d(x, LA_volume_area_4ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volume_area_4ch_smooth = yy_sg

        np.savetxt(os.path.join(
            results_dir, 'LA_volume_SR_4Ch.txt'), LA_volume_SR_4Ch)
        np.savetxt(os.path.join(
            results_dir, 'LA_volume_SR_4Ch_smooth.txt'),
            LA_volume_SR_4Ch_smooth)
        np.savetxt(os.path.join(
            results_dir, 'LA_volume_area_4ch.txt'), LA_volume_area_4ch)
        np.savetxt(os.path.join(
            results_dir, 'LA_volume_area_4ch_smooth.txt'),
            LA_volume_area_4ch_smooth)
    else:
        LA_volume_area_4ch_smooth = -1*np.ones(N_frames_4Ch)
        LA_volume_SR_4Ch_smooth = -1*np.ones(N_frames_4Ch)

    # RA volumes 
    if QC_atria_4Ch_RA == 0:
        RA_volumes_SR = -1*np.ones(N_frames_4Ch)
        RA_volumes_area = -1*np.ones(N_frames_4Ch) 
        for fr in range(N_frames_4Ch):
            d1d2 = la_diams_RV[fr, :] * la_diams_RV[fr, :]
            length = length_top_RV[fr]
            RA_volumes_SR[fr] = math.pi / 4 * length * \
                np.sum(d1d2) / Nsegments_length / 1000
            RA_volumes_area[fr] = 0.85 * area_RA[fr] * \
                area_RA[fr] / length / 1000
        
        x = np.linspace(0, N_frames_4Ch - 1, N_frames_4Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch)
        itp = interp1d(x, RA_volumes_SR)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        RA_volumes_SR_smooth = yy_sg
        itp = interp1d(x, RA_volumes_area)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        RA_volumes_area_smooth = yy_sg

        np.savetxt(os.path.join(
            results_dir, 'RA_volumes_SR.txt'), RA_volumes_SR)
        np.savetxt(os.path.join(
            results_dir, 'RA_volumes_area.txt'), RA_volumes_area)
        np.savetxt(os.path.join(
            results_dir, 'RA_volumes_SR_smooth.txt'), RA_volumes_SR_smooth)
        np.savetxt(os.path.join(
            results_dir, 'RA_volumes_area_smooth.txt'), RA_volumes_area_smooth)
    else:
        RA_volumes_area_smooth = -1*np.ones(N_frames_4Ch)
        RA_volumes_SR_smooth = -1*np.ones(N_frames_4Ch)

    # =====================================================================
    # Compute volume by combining 2ch and 4ch views
    # =====================================================================
    if QC_atria_4Ch_LA == 0 and QC_atria_2Ch == 0 and N_frames_2Ch == N_frames_4Ch:
        LA_volume_combined_SR = -1*np.ones(N_frames_4Ch)
        LA_volume_combined_area = -1*np.ones(N_frames_4Ch)
        for fr in range(N_frames_4Ch):
            # Combined volume based on Simpson's rule
            d1d2 = la_diams_2Ch[fr, :] * la_diams_4Ch[fr, :]
            length = np.min([length_top_2Ch[fr], length_top_4Ch[fr]])
            LA_volume_combined_SR[fr] = math.pi / 4 * length * \
                np.sum(d1d2) / Nsegments_length / 1000
            
            # Combined volume based on number of pixels
            LA_volume_combined_area[fr] = 0.85 * area_LA_2Ch[fr] * \
            area_LA_4Ch[fr] / length / 1000
            
        x = np.linspace(0, N_frames_4Ch - 1, N_frames_4Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch)
        itp = interp1d(x, LA_volume_combined_SR)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volume_combined_SR_smooth = yy_sg
        
        itp = interp1d(x, LA_volume_combined_area)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volume_combined_area_smooth = yy_sg

        # =====================================================================
        # Calculate combined strain using 2ch and 4ch views
        # =====================================================================
        LA_strain_circum_combined = LA_strain_circum_4Ch_smooth +\
            LA_strain_circum_2Ch_smooth / 2
        LA_strain_longitud_combined = LA_strain_longitud_4Ch_smooth +\
            LA_strain_longitud_2ch_smooth / 2
        LA_strain_circum_combined = smooth_curve_1d(LA_strain_circum_combined, N_frames_4Ch*p, s)
        LA_strain_longitud_combined = smooth_curve_1d(LA_strain_longitud_combined, N_frames_4Ch*p, s)
        
        LA_strainRate_circum_combined = LA_strainRate_circum_4Ch_smooth +\
            LA_strainRate_circum_2Ch_smooth / 2
        LA_strainRate_longitud_combined = LA_strainRate_longitud_4Ch_smooth +\
            LA_strainRate_longitud_2ch_smooth / 2

        np.savetxt(os.path.join(results_dir, 'LA_volume_combined_SR.txt'),
                   LA_volume_combined_SR)
        np.savetxt(os.path.join(results_dir, 'LA_volume_combined_area.txt'),
                   LA_volume_combined_area)
        np.savetxt(os.path.join(results_dir,
                                'LA_volume_combined_SR_smooth.txt'),
                   LA_volume_combined_SR_smooth)
        np.savetxt(os.path.join(results_dir,
                                'LA_volume_combined_area_smooth.txt'),
                   LA_volume_combined_area_smooth)
        np.savetxt(os.path.join(results_dir, 'LA_strain_circum_combined.txt'),
                   LA_strain_circum_combined)
        np.savetxt(os.path.join(results_dir, 'LA_strain_longitud_combined.txt'),
                   LA_strain_longitud_combined)
        np.savetxt(os.path.join(results_dir, 'LA_strainRate_circum_combined.txt'),
                   LA_strainRate_circum_combined)
        np.savetxt(os.path.join(results_dir, 'LA_strainRate_longitud_combined.txt'),
                   LA_strainRate_longitud_combined)

    # =========================================================================
    # Compute params if not the same number of slices between views
    # =========================================================================
    elif QC_atria_4Ch_LA == 0 and QC_atria_2Ch == 0 and N_frames_2Ch != N_frames_4Ch:
        max_frames = max(N_frames_2Ch, N_frames_4Ch)
        LA_volume_combined_SR = -1*np.ones(N_frames_4Ch)
        LA_volume_combined_area = -1*np.ones(N_frames_4Ch)
        length_top_2Ch_itp = resample(length_top_2Ch, max_frames)
        length_top_4Ch_itp = resample(length_top_4Ch, max_frames)
        area_LA_2Ch_itp = resample(area_LA_2Ch, max_frames)
        area_LA_4Ch_itp = resample(area_LA_4Ch, max_frames)
        la_diams_2Ch_itp = resample(la_diams_2Ch, max_frames)
        la_diams_4Ch_itp = resample(la_diams_4Ch, max_frames)

        for fr in range(max_frames):
            # Simpson's rule
            d1d2 = la_diams_2Ch_itp[fr, :] * la_diams_4Ch_itp[fr, :]
            length = np.min([length_top_2Ch_itp[fr], length_top_4Ch_itp[fr]])
            LA_volume_combined_SR[fr] = math.pi / 4 * length * \
                np.sum(d1d2) / Nsegments_length / 1000

            # Pixel method
            if N_frames_2Ch == N_frames_4Ch:
                LA_volume_combined_area[fr] = 0.85 * area_LA_2Ch_itp[fr] * \
                    area_LA_4Ch_itp[fr] / length / 1000

        x = np.linspace(0, max_frames - 1, max_frames)
        xx = np.linspace(np.min(x), np.max(x), max_frames)
        itp = interp1d(x, LA_volume_combined_SR)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volume_combined_SR_smooth = yy_sg

        itp = interp1d(x, LA_volume_combined_area)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volume_combined_area_smooth = yy_sg
        LA_strain_circum_combined = -1*np.ones(20)
        LA_strain_longitud_combined = -1*np.ones(20)
        LA_strainRate_circum_combined = -1*np.ones(20)
        LA_strainRate_longitud_combined = -1*np.ones(20)

        np.savetxt(os.path.join(results_dir, 'LA_volume_combined_SR.txt'),
                   LA_volume_combined_SR)
        np.savetxt(os.path.join(results_dir, 'LA_volume_combined_area.txt'),
                   LA_volume_combined_area)
        np.savetxt(os.path.join(results_dir,
                                'LA_volume_combined_SR_smooth.txt'),
                   LA_volume_combined_SR_smooth)
        np.savetxt(os.path.join(results_dir,
                                'LA_volume_combined_area_smooth.txt'),
                   LA_volume_combined_area_smooth)
    else:
        LA_volume_combined_SR_smooth = -1*np.ones(N_frames_4Ch)
        LA_volume_combined_area_smooth = -1*np.ones(N_frames_4Ch)
        LA_strain_circum_combined = -1*np.ones(N_frames_4Ch)
        LA_strain_longitud_combined = -1*np.ones(N_frames_4Ch)
        LA_strainRate_circum_combined = -1*np.ones(N_frames_4Ch)
        LA_strainRate_longitud_combined = -1*np.ones(N_frames_4Ch)

    # =========================================================================
    # Peak volume
    # =========================================================================
    peak_LA_volume_area_2ch = LA_volume_area_2ch_smooth.max()
    peak_LA_volume_SR_2ch = LA_volume_SR_2ch_smooth.max()
    peak_LA_volume_area_4ch = LA_volume_area_4ch_smooth.max()
    peak_LA_volume_SR_4ch = LA_volume_SR_4Ch_smooth.max()
    peak_LA_volume_area_combined = LA_volume_combined_area_smooth.max()
    peak_LA_volume_SR_combined = LA_volume_combined_SR_smooth.max()
    peak_RA_volume_area = RA_volumes_area_smooth.max()
    peak_RA_volume_SR = RA_volumes_SR_smooth.max()

    # =========================================================================
    # Peak strain
    # =========================================================================
    peak_LA_strain_circum_2ch_max = LA_strain_circum_2Ch_smooth.max()
    peak_LA_strain_longitud_2ch_max = LA_strain_longitud_2ch_smooth.max()
    peak_LA_strain_circum_4ch_max = LA_strain_circum_4Ch_smooth.max()
    peak_LA_strain_longitud_4ch_max = LA_strain_longitud_4Ch_smooth.max()
    peak_LA_strain_circum_combined_max = LA_strain_circum_combined.max()
    peak_LA_strain_longitud_combined_max = LA_strain_longitud_combined.max()
    peak_RA_strain_circum_max = RA_strain_circum_4Ch_smooth.max()
    peak_RA_strain_longitud_max = RA_strain_longitud_4Ch_smooth.max()

    ES_frame = LA_volume_combined_SR_smooth.argmax() * p
    peak_LA_strain_circum_2ch_ES = LA_strain_circum_2Ch_smooth[ES_frame]
    peak_LA_strain_longitud_2ch_ES = LA_strain_longitud_2ch_smooth[ES_frame]
    peak_LA_strain_circum_4ch_ES = LA_strain_circum_4Ch_smooth[ES_frame]
    peak_LA_strain_longitud_4ch_ES = LA_strain_longitud_4Ch_smooth[ES_frame]
    peak_LA_strain_circum_combined_ES = LA_strain_circum_combined[ES_frame]
    peak_LA_strain_longitud_combined_ES = LA_strain_longitud_combined[ES_frame]

    ES_frame_RA = RA_volumes_SR_smooth.argmax() * p
    peak_RA_strain_circum_ES = RA_strain_circum_4Ch_smooth[ES_frame_RA]
    peak_RA_strain_longitud_ES = RA_strain_longitud_4Ch_smooth[ES_frame_RA]

    first10_2ch = int(len(LA_strainRate_circum_2Ch_smooth)*0.1)
    first10_4ch = int(len(LA_strainRate_circum_4Ch_smooth)*0.1)
    peak_LA_strainRate_circum_2ch = LA_strainRate_circum_2Ch_smooth[first10_2ch:].max()
    peak_LA_strainRate_longit_2ch = LA_strainRate_longitud_2ch_smooth[first10_2ch:].max()
    peak_LA_strainRate_circum_4ch = LA_strainRate_circum_4Ch_smooth[first10_4ch:].max()
    peak_LA_strainRate_longitud_4ch = LA_strainRate_longitud_4Ch_smooth[first10_4ch:].max()
    peak_LA_strainRate_circum_combo = LA_strainRate_circum_combined[first10_4ch:].max()
    peak_LA_strainRate_longitud_combo = LA_strainRate_longitud_combined[first10_4ch:].max()
    peak_RA_strainRate_circum = RA_strainRate_circum_4Ch_smooth[first10_4ch:].max()
    peak_RA_strainRate_longitud = RA_strainRate_longitud_4Ch_smooth[first10_4ch:].max()

    # =========================================================================
    # PLOTS
    # =========================================================================
    plt.figure()
    plt.plot(tt_2ch_orig, LA_volume_SR_2ch_smooth, label='Simpson - 2Ch')
    plt.plot(tt_4ch_orig, LA_volume_SR_4Ch_smooth, label='Simpson - 4Ch')
    plt.plot(tt_2ch_orig, LA_volume_area_2ch_smooth, label='Area - 2Ch')
    plt.plot(tt_4ch_orig, LA_volume_area_4ch_smooth, label='Area - 4Ch')
    plt.plot(tt_4ch_orig, LA_volume_combined_SR_smooth, label='Simpson - combined')
    plt.plot(tt_4ch_orig, LA_volume_combined_area_smooth, label='Area method - combined')
    plt.legend()
    plt.xlabel('Trigger Time (ms)')
    plt.ylabel('Volume (mL/m2)')
    plt.title('Left Atrial Volume')
    plt.savefig(os.path.join(results_dir, 'LA_volume.png'))
    plt.close('all')

    plt.figure()
    plt.plot(tt_4ch_orig, RA_volumes_SR_smooth, label='Simpson method')
    plt.plot(tt_4ch_orig, RA_volumes_area_smooth, label='Area method')
    plt.legend()
    plt.xlabel('Trigger Time (ms)')
    plt.ylabel('Volume (mL/m2)')
    plt.title('Right Atrial Volume')
    plt.savefig(os.path.join(results_dir, 'RA_volume.png'))
    plt.close('all')

    # Interpolation
    if debug:
        interp_folder = '/media/ec17/WINDOWS_DATA/Flow_project/Atrial_strain/log/interp'
        if not os.path.exists(interp_folder):
            os.mkdir(interp_folder)
        x_2ch = np.arange(N_frames_2Ch)
        x_4ch = np.arange(N_frames_4Ch)
        plt.figure()
        plt.plot(LA_volume_SR_2ch, label='Original')
        plt.scatter(x_2ch, LA_volume_SR_2ch)
        plt.plot(LA_volume_SR_2ch_smooth, 'r', label='Smoothed')
        plt.title('LA volume Simpson 2ch')
        plt.legend()
        plt.savefig(os.path.join(interp_folder,
                    f'LA_vol_Simpson2ch_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(LA_volume_SR_4Ch, label='Original')
        plt.scatter(x_4ch, LA_volume_SR_4Ch)
        plt.plot(LA_volume_SR_4Ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA volume Simpson 4ch')
        plt.savefig(os.path.join(interp_folder,
                    f'LA_vol_Simpson4ch_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(LA_volume_area_2ch, label='Original')
        plt.scatter(x_2ch, LA_volume_area_2ch)
        plt.plot(LA_volume_area_2ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA volume area 2ch')
        plt.savefig(os.path.join(interp_folder, f'LA_vol_Area2ch_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(LA_volume_area_4ch, label='Original')
        plt.scatter(x_4ch, LA_volume_area_4ch)
        plt.plot(LA_volume_area_4ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA volume area 4ch')
        plt.savefig(os.path.join(interp_folder, f'LA_vol_Area4ch_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(LA_volume_combined_SR, label='Original')
        plt.scatter(x_2ch, LA_volume_combined_SR)
        plt.plot(LA_volume_combined_SR_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA volume Simpson combined')
        plt.savefig(os.path.join(interp_folder,
                    f'LA_vol_SimpsonCombined_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(LA_volume_combined_area, label='Original')
        plt.scatter(x_4ch, LA_volume_combined_area)
        plt.plot(LA_volume_combined_area_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA volume area combined')
        plt.savefig(os.path.join(interp_folder,
                    f'LA_vol_AreaCombined_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(RA_volumes_SR, label='Original')
        plt.scatter(x_4ch, RA_volumes_SR)
        plt.plot(RA_volumes_SR_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('RA volume SR')
        plt.savefig(os.path.join(interp_folder,
                    f'RA_vol_SR_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(RA_volumes_area, label='Original')
        plt.scatter(x_4ch, RA_volumes_area)
        plt.plot(RA_volumes_area_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('RA volume area')
        plt.savefig(os.path.join(interp_folder,
                    f'RA_vol_Area_{study_ID}.png'))
        plt.close('all')

    # Circumferential strain
    plt.figure()
    plt.plot(tt_2ch, LA_strain_circum_2Ch_smooth, label='2ch')
    plt.plot(tt_2ch[LA_strain_circum_2Ch_smooth.argmax()],
                peak_LA_strain_circum_2ch_max, 'ro', label='Peak strain - max')
    plt.plot(
        tt_2ch[ES_frame], peak_LA_strain_circum_2ch_ES, 'b*', label='Peak strain - ES')
    plt.plot(tt_4ch, LA_strain_circum_4Ch_smooth, 'k', label='4ch')
    plt.plot(tt_4ch[LA_strain_circum_4Ch_smooth.argmax()],
                peak_LA_strain_circum_4ch_max, 'ro')
    plt.plot(tt_4ch[ES_frame], peak_LA_strain_circum_4ch_ES, 'b*')
    plt.plot(tt_4ch, LA_strain_circum_combined, label='Combined')
    plt.plot(tt_4ch[LA_strain_circum_combined.argmax()],
                peak_LA_strain_circum_combined_max, 'ro')
    plt.plot(tt_4ch[ES_frame], peak_LA_strain_circum_combined_ES, 'b*')
    plt.xlabel('Trigger Time (ms)')
    plt.ylabel('Strain')
    plt.legend()
    plt.title('LA Circumferential Strain')
    plt.savefig(os.path.join(results_dir, 'LA_circum_strain.png'))
    plt.close('all')

    # Circumferential strain rate
    plt.figure()
    plt.plot(tt_2ch, LA_strainRate_circum_2Ch_smooth, label='2ch')
    plt.plot(tt_4ch, LA_strainRate_circum_4Ch_smooth, label='4ch')
    plt.plot(tt_4ch, LA_strainRate_circum_combined, label='combined')
    plt.plot(tt_2ch[LA_strainRate_circum_2Ch_smooth[first10_2ch:].argmax() + first10_2ch], peak_LA_strainRate_circum_2ch, 'ro')
    plt.plot(tt_4ch[LA_strainRate_circum_4Ch_smooth[first10_4ch:].argmax() + first10_4ch], peak_LA_strainRate_circum_4ch, 'bo')
    plt.plot(tt_4ch[LA_strainRate_circum_combined[first10_4ch:].argmax() + first10_4ch], peak_LA_strainRate_circum_combo, 'ko')
    plt.legend()
    plt.xlabel('Trigger Time (ms)')
    plt.ylabel('Strain Rate')
    plt.title('LA Circumferntial Strain Rate')
    plt.savefig(os.path.join(results_dir, 'LA_circum_strainRate.png'))
    plt.close('all')

    # Longitudinal strain
    plt.figure()
    plt.plot(tt_2ch, LA_strain_longitud_2ch_smooth, label='2ch')
    plt.plot(tt_2ch[LA_strain_longitud_2ch_smooth.argmax()],
                peak_LA_strain_longitud_2ch_max, 'ro',
                label='Peak strain - max')
    plt.plot(tt_2ch[ES_frame], peak_LA_strain_longitud_2ch_ES,
                'b*', label='Peak strain - ES')
    plt.plot(tt_4ch, LA_strain_longitud_4Ch_smooth, label='4ch')
    plt.plot(tt_4ch[LA_strain_longitud_4Ch_smooth.argmax()],
                peak_LA_strain_longitud_4ch_max, 'ro')
    plt.plot(tt_4ch[ES_frame], peak_LA_strain_longitud_4ch_ES, 'b*')
    plt.plot(tt_4ch, LA_strain_longitud_combined, label='Combined')
    plt.plot(tt_4ch[LA_strain_longitud_combined.argmax()],
                peak_LA_strain_longitud_combined_max, 'ro')
    plt.plot(tt_4ch[ES_frame], peak_LA_strain_longitud_combined_ES, 'b*')
    plt.legend()
    plt.xlabel('Trigger Time (ms)')
    plt.ylabel('Strain')
    plt.title('LA Longitudinal Strain')
    plt.savefig(os.path.join(results_dir, 'LA_longitud_strain.png'))
    plt.close('all')

    # Longitudinal strain rate
    plt.figure()
    plt.plot(tt_2ch, LA_strainRate_longitud_2ch_smooth, label='2ch')
    plt.plot(tt_4ch, LA_strainRate_longitud_4Ch_smooth, label='4ch')
    plt.plot(tt_4ch, LA_strainRate_longitud_combined, label='combined')
    plt.plot(tt_2ch[LA_strainRate_longitud_2ch_smooth[first10_2ch:].argmax() + first10_2ch], peak_LA_strainRate_longit_2ch, 'ro')
    plt.plot(tt_4ch[LA_strainRate_longitud_4Ch_smooth[first10_4ch:].argmax() + first10_4ch], peak_LA_strainRate_longitud_4ch, 'bo')
    plt.plot(tt_4ch[LA_strainRate_longitud_combined[first10_4ch:].argmax() + first10_4ch], peak_LA_strainRate_longitud_combo, 'ko')
    plt.legend()
    plt.xlabel('Trigger Time (ms)')
    plt.ylabel('Strain Rate')
    plt.title('LA Longitudinal Strain Rate')
    plt.savefig(os.path.join(results_dir, 'LA_longit_strainRate.png'))
    plt.close('all')

    # RA strain
    plt.figure()
    plt.plot(tt_4ch, RA_strain_circum_4Ch_smooth, label='Circum')
    plt.plot(tt_4ch[RA_strain_circum_4Ch_smooth.argmax()],
                peak_RA_strain_circum_max, 'ro', label='Peak strain - max')
    plt.plot(tt_4ch[ES_frame_RA], peak_RA_strain_circum_ES, 'b*',
                label='Peak strain - ES')
    plt.plot(tt_4ch, RA_strain_longitud_4Ch_smooth, label='Longitud')
    plt.plot(tt_4ch[RA_strain_longitud_4Ch_smooth.argmax()],
                peak_RA_strain_longitud_max, 'ro')
    plt.plot(tt_4ch[ES_frame_RA], peak_RA_strain_longitud_ES, 'b*')
    plt.legend()
    plt.xlabel('Trigger Time (ms)')
    plt.ylabel('Strain')
    plt.title('RA Strain')
    plt.savefig(os.path.join(results_dir, 'RA_strain.png'))
    plt.close('all')

    # RA Strain Rate
    plt.figure()
    plt.plot(tt_4ch, RA_strainRate_circum_4Ch_smooth, label='circum')
    plt.plot(tt_4ch, RA_strainRate_longitud_4Ch_smooth, label='longitud')
    plt.plot(tt_4ch[RA_strainRate_circum_4Ch_smooth[first10_4ch:].argmax() + first10_4ch], peak_RA_strainRate_circum, 'bo')
    plt.plot(tt_4ch[RA_strainRate_longitud_4Ch_smooth[first10_4ch:].argmax() + first10_4ch], peak_RA_strainRate_longitud, 'ko')
    plt.legend()
    plt.xlabel('Trigger Time (ms)')
    plt.ylabel('Strain Rate')
    plt.title('RA Strain Rate')
    plt.savefig(os.path.join(results_dir, 'RA_strainRate.png'))
    plt.close('all')

    # Interpolate
    if debug:
        plt.figure()
        plt.plot(tt_2ch_orig, LA_strain_circum_2Ch, label='Original')
        plt.scatter(tt_2ch_orig, LA_strain_circum_2Ch)
        plt.plot(tt_2ch, LA_strain_circum_2Ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA strain circum 2ch')
        plt.savefig(os.path.join(interp_folder,
                    f'LA_strain_circum2ch_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(tt_4ch_orig, LA_strain_circum_4Ch, label='Original')
        plt.scatter(tt_4ch_orig, LA_strain_circum_4Ch)
        plt.plot(tt_4ch, LA_strain_circum_4Ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA strain circum 4ch')
        plt.savefig(os.path.join(interp_folder,
                    f'LA_strain_circum4ch_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(tt_2ch_orig, LA_strain_longitud_2ch, label='Original')
        plt.scatter(tt_2ch_orig, LA_strain_longitud_2ch)
        plt.plot(tt_2ch, LA_strain_longitud_2ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA strain longitud 2ch')
        plt.savefig(os.path.join(interp_folder,
                    f'LA_strain_longitud2ch_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(tt_4ch_orig, LA_strain_longitud_4Ch, label='Original')
        plt.scatter(tt_4ch_orig, LA_strain_longitud_4Ch)
        plt.plot(tt_4ch, LA_strain_longitud_4Ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('LA strain longitud 4ch')
        plt.savefig(os.path.join(interp_folder,
                    f'LA_strain_longitud4ch_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(tt_4ch_orig, RA_strain_circum_4Ch, label='Original')
        plt.scatter(tt_4ch_orig, RA_strain_circum_4Ch)
        plt.plot(tt_4ch, RA_strain_circum_4Ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('RA strain circum')
        plt.savefig(os.path.join(interp_folder,
                    f'RA_strain_circum_{study_ID}.png'))
        plt.close('all')

        plt.figure()
        plt.plot(tt_4ch_orig, RA_strain_longitud_4Ch, label='Original')
        plt.scatter(tt_4ch_orig, RA_strain_longitud_4Ch)
        plt.plot(tt_4ch, RA_strain_longitud_4Ch_smooth, 'r', label='Smoothed')
        plt.legend()
        plt.title('RA strain longitud')
        plt.savefig(os.path.join(interp_folder,
                    f'RA_strain_longitud_{study_ID}.png'))
        plt.close('all')

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    vols = -1*np.ones(33, dtype=object)
    vols[0] = study_ID
    vols[1] = peak_LA_volume_area_combined
    vols[2] = peak_LA_volume_SR_combined
    vols[3] = peak_LA_volume_area_2ch
    vols[4] = peak_LA_volume_SR_2ch
    vols[5] = peak_LA_volume_area_4ch
    vols[6] = peak_LA_volume_SR_4ch

    vols[7] = peak_LA_strain_circum_combined_ES
    vols[8] = peak_LA_strain_circum_combined_max
    vols[9] = peak_LA_strain_circum_2ch_ES
    vols[10] = peak_LA_strain_circum_2ch_max
    vols[11] = peak_LA_strain_circum_4ch_ES
    vols[12] = peak_LA_strain_circum_4ch_max

    vols[13] = peak_LA_strain_longitud_combined_ES
    vols[14] = peak_LA_strain_longitud_combined_max
    vols[15] = peak_LA_strain_longitud_2ch_ES
    vols[16] = peak_LA_strain_longitud_2ch_max
    vols[17] = peak_LA_strain_longitud_4ch_ES
    vols[18] = peak_LA_strain_longitud_4ch_max

    vols[19] = peak_RA_volume_area
    vols[20] = peak_RA_volume_SR
    vols[21] = peak_RA_strain_circum_ES
    vols[22] = peak_RA_strain_circum_max
    vols[23] = peak_RA_strain_longitud_ES
    vols[24] = peak_RA_strain_longitud_max

    vols[25] = peak_LA_strainRate_circum_combo
    vols[26] = peak_LA_strainRate_circum_2ch
    vols[27] = peak_LA_strainRate_circum_4ch
    vols[28] = peak_LA_strainRate_longitud_combo
    vols[29] = peak_LA_strainRate_longit_2ch
    vols[30] = peak_LA_strainRate_longitud_4ch
    vols[31] = peak_RA_strainRate_circum
    vols[32] = peak_RA_strainRate_longitud

    vols = np.reshape(vols, [1, 33])
    df = pd.DataFrame(vols)
    header = ['eid', 'LA_vol_area_combo', 'LA_vol_SR_combo', 'LA_vol_area_2ch', 'LA_vol_SR_2ch',
                'LA_vol_area_4ch', 'LA_vol_SR_4ch',  'LA_strain_circum_combined_ES',
                'LA_strain_circum_combined_max', 'LA_strain_circum_2ch_ES',
                'LA_strain_circum_2ch_max', 'LA_strain_circum_4ch_ES',
                'LA_strain_circum_4ch_max',
                'LA_strain_long_2ch_ES', 'LA_strain_long_2ch_max',
                'LA_strain_long_4ch_ES', 'LA_strain_long_4ch_max',
                'LA_strain_long_combo_ES', 'LA_strain_long_combo_max',
                'RA_volume_area', 'RA_volume_SR', 'RA_strain_circum_ES',
                'RA_strain_circum_max', 'RA_strain_long_ES',
                'RA_strain_long_max', 'LA_strainRate_circum_combo', 'LA_strainRate_circum_2ch',
                'LA_strainRate_circum_4ch', 'LA_strainRate_longit_combo',
                'LA_strainRate_longit_2ch', 'LA_strainRate_longit_4ch',
                'RA_strainRate_circum', 'RA_strainRate_longit'
                ]
    df.to_csv(os.path.join(results_dir, 'atrial_peak_params.csv'),
              header=header, index=False)
    
    return df, header
