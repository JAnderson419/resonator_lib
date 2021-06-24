# %%
import gdspy as gd
import numpy as np
from enum import Enum
from typing import Union

class IDT_Type(Enum):
    STANDARD = 'standard'
    EWC = 'EWC'
    DART = 'dart'
    SPLIT_FINGER = 'split'
    FOCUSED = 'focused'


def gsg_pad(lib, layers=[0,1], p=150, x=80, y=70, connect_grounds=False):
    c = lib.new_cell(f'GSG_pad_P{p}_x{x}_y{y}')
    #  signal pad
    c.add(gd.Rectangle((-x/2, -y/2), (x/2, y/2), layer=layers['M2']))
    # rt = gd.Polygon([(x/2, -y/2),(x/2+40, -y/2+10),(x/2+40, y/2-10),(x/2, y/2)])

    #  ground pads
    c.add(gd.Rectangle((-x/2-20, -y/2-p), (x/2, y/2-p), layer=layers['M2']))
    c.add(gd.Rectangle((-x/2-20, -y/2+p), (x/2, y/2+p), layer=layers['M2']))

    if connect_grounds:
        c.add(gd.Rectangle((-x/2-40, -y/2-p), (-x/2-20, y/2+p), layer=layers['M2']))
    return c

def bar_resonator(lib, pads, layers, d=150, x=30, y=200, yoff=40, trench=False):
    """

    :param lib:
    :param pads: gdspy padset cell
    :param layers: list of mask layers in format [trench, metal1]
    :param d: distance between port 1 and 2 electrodes, inner edge to inner edge
    :param x: width of electrodes
    :param y: length of overlap between port 1 and 2 electrodes
    :param yoff: additional length of electrode before pad taper
    :param trench: bool, determines if resonator has a trench or not
    :return: c: bar resonator cell
    """
    c = lib.new_cell(f'bar_res_d{d}_x{x}_y{y}')
    c.add(gd.Rectangle((-d/2-x, -y/2-yoff), (-d/2, y/2), layers['M1']))
    c.add(gd.Rectangle((d / 2, -y / 2), (d / 2 + x, y / 2+yoff), layers['M1']))
    pad_extent = gd.CellReference(pads).get_bounding_box()
    c.add(gd.CellReference(pads,
                           rotation=-90,
                           origin=(d / 2 + x/2, y / 2 + pad_extent[1][0]+30+yoff)))
    c.add(gd.Polygon([(d / 2, y / 2+yoff),
                      (d / 2 + (x-70)/2, y / 2 + 30+yoff),
                      (d / 2 + x - (x-70)/2, y / 2+30+yoff),
                      (d / 2 + x, y / 2+yoff)
                      ], layers[1]))
    c.add(gd.CellReference(pads,
                           rotation=90,
                           origin=(-d / 2 - x / 2, -y / 2 - pad_extent[1][0]-30-yoff)))
    c.add(gd.Polygon([(-d / 2, -y / 2-yoff),
                      (-d / 2 - (x - 70) / 2, -y / 2 - 30-yoff),
                      (-d / 2 - x + (x - 70) / 2, -y / 2 - 30-yoff),
                      (-d / 2 - x, -y / 2-yoff)
                      ], layers['M1']))


    if trench and d>10:
        c.add(gd.Rectangle((-d/2+5, -y/2-yoff/2), (d/2-5, y/2+yoff/2), layers['Trench']))
    extent = c.get_bounding_box()
    c.add(gd.Text(f'bar_res_d{d}_x{x}_y{y}', 20, (extent[0][0], extent[1][1]+5), layers['M1']))
    return c


def bar_resonator_matrix(lib, pads, layers, ds, ys, trench=False):
    m = lib.new_cell(f'Resonator Matrix')
    max_x = 0
    max_y = 0
    cells = []
    x0 = []
    y0 = []
    for i, d in enumerate(ds):
        for j, y in enumerate(ys):
            c = bar_resonator(lib, pads, layers, d=d, y=y, trench=trench)
            x0.append(i)
            y0.append(j)
            cells.append(c)
            extent = c.get_bounding_box()
            if extent[1][0] - extent[0][0] > max_x:
                max_x = extent[1][0] - extent[0][0]
            else:
                pass
            if extent[1][1] - extent[0][1] > max_y:
                max_y = extent[1][1] - extent[0][1]
            else:
                pass
    x0 = (np.asarray(x0) * max_x * 1.1)
    y0 = (np.asarray(y0) * max_y * 1.1)
    for i, r in enumerate(cells):
        m.add(gd.CellReference(r, origin=(x0[i], y0[i])))


def idt_device(lib: gd.GdsLibrary, pads: gd.Cell, layers: dict, lmda: float, g_idt: float, idt_type: IDT_Type=IDT_Type.STANDARD,
               process_bias: float=0, n_idt: int=20, w_b: float=20, s_b: float=1, l_idt: float=None,
               reflector: bool=False, g_r: float=None, w_br: float=None, n_idtr: int=None, w_r: float=None,
               s_r: float=None, theta: float=None) -> gd.Cell:

    dev_x_extent = 0
    # Parameter Setup
    if l_idt is None:
        l_idt = 100*lmda
    if idt_type == IDT_Type.STANDARD:
        cell_name = 'dIDT'
        w_idt = lmda / 4 + process_bias
        s_idt = lmda / 4 - process_bias
        idts = idt_cell(lib, layers, w_idt, s_idt, l_idt, n_idt, w_b, s_b)
    elif idt_type == IDT_Type.EWC:
        if process_bias != 0:
            raise NotImplementedError("Process bias not implemented for SPUDT devices")
        cell_name = 'dEWC'
        idts = ewc_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
    elif idt_type == IDT_Type.SPLIT_FINGER:
        cell_name = 'dSplitFing'
        idts = split_finger_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
    elif idt_type == IDT_Type.DART:
        if process_bias != 0:
            raise NotImplementedError("Process bias not implemented for SPUDT devices")
        cell_name = 'dDART'
        idts = dart_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
    elif idt_type == IDT_Type.FOCUSED:
        if theta is None:
            raise ValueError("No angle defined for focused IDT.")
        else:
            cell_name = 'dFoc'
            w_idt = lmda / 4 + process_bias
            s_idt = lmda / 4 - process_bias
            idts = focused_idt_cell(lib, layers, w_idt, s_idt, theta, g_idt, n_idt, w_b, s_b)
    cell_name = cell_name+f'_lambda{lmda:.1f}_g{g_idt:.1f}_nIDT{n_idt:.1f}_IDTl{l_idt:.1f}_wb{w_b:.1f}_sb{s_b:.1f}_bias{process_bias:.1f}'
    if reflector is True:
        if g_r is None: g_r = 5*lmda # TODO: Change default value based on simulated performance
        if w_br is None: w_br = w_b
        if w_r is None: w_r = lmda / 4 + process_bias
        if s_r is None: s_r = lmda / 4 - process_bias
        if n_idtr is None: n_idtr = n_idt
        cell_name = cell_name+f'_gr{g_r:.1f}_wr{w_r:.1f}_sr{s_r:.1f}_wbr{w_br:.1f}_nR{n_idtr:.1f}'

    # Cell Creation
    c = lib.new_cell(cell_name)
    idt_extent = idts.get_bounding_box()
    if idt_type == IDT_Type.FOCUSED:
        idt_offset = 0  # cell extent goes to focal point
    else:
        idt_offset = g_idt/2 + idt_extent[1][0]
    c.add(gd.CellReference(idts, origin=(-idt_offset, 0), rotation=180)) # rotation puts idt finger at gap edge
    c.add(gd.CellReference(idts, origin=(idt_offset, 0), rotation=0))
    dev_x_extent = idt_offset + idt_extent[1][0]
    if reflector is True:
        if idt_type == IDT_Type.FOCUSED:
            r = focused_idt_reflector(lib, layers, w_r, s_r, theta, g_idt+2*(g_r+n_idt*lmda), n_idtr, w_br)
            r_extent = gd.CellReference(r).get_bounding_box()
            r_offset = 0  # r_extent[0][0]
            dev_x_extent = r_extent[1][0]
        else:
            r = idt_reflector(lib, layers, w_r, s_r, l_idt, n_idtr, w_br)
            r_extent = gd.CellReference(r).get_bounding_box()
            r_offset = g_idt/2 + (idt_extent[1][0]-idt_extent[0][0]) + g_r + r_extent[1][0]
            dev_x_extent = dev_x_extent + r_extent[1][0]
        c.add(gd.CellReference(r, origin=(r_offset, 0)))
        c.add(gd.CellReference(r, origin=(-r_offset, 0), rotation=180))


    # generate pads
    pad_extent = pads.get_bounding_box()
    pad_tapery = 50  # offset distance for tapering from pad edge to IDT busbar
    v1_y = 20
    pad_yoffset = idt_extent[1][1]+pad_extent[1][0]+pad_tapery+ v1_y
    if idt_type == IDT_Type.FOCUSED:
        pad_xoffset = (g_idt/2 + n_idt*lmda)*np.cos(theta*np.pi/360) - n_idt*lmda/2 - w_b*np.sin(theta*np.pi/360)
    else:
        pad_xoffset = idt_offset
    for (i, j) in [(-1, 1), (1, -1)]:
        c.add(gd.CellReference(pads,
                               rotation=i*90,
                               origin=(i*pad_xoffset, j*pad_yoffset)))
        # signal connection to device
        c.add(gd.Polygon([(i*(pad_xoffset -n_idt * lmda/2), j*(idt_extent[1][1] + v1_y)),
                          (i*(pad_xoffset-34), j*(idt_extent[1][1] + v1_y+pad_tapery)),
                          (i*(pad_xoffset+34), j*(idt_extent[1][1] + v1_y+pad_tapery)),
                          (i*(pad_xoffset + n_idt*lmda/2), j*(idt_extent[1][1] + v1_y)),
                          ], layers['M2']))
        c.add(gd.Rectangle((i*(pad_xoffset-n_idt * lmda/2), j*(idt_extent[1][1])),
                           (i*(pad_xoffset+n_idt * lmda/2), j*(idt_extent[1][1] + v1_y)),
                           layers['M2']))
        c.add(gd.Rectangle((i*(pad_xoffset-n_idt * lmda/2), j*(idt_extent[1][1])),
                           (i*(pad_xoffset+n_idt * lmda/2), j*(idt_extent[1][1] + v1_y)),
                           layers['M1']))

        # ground pad to pad connections
        if pad_xoffset+(pad_extent[1][1]-pad_extent[0][1])/2-70 > dev_x_extent+5:
            ground_x_offset = pad_xoffset+(pad_extent[1][1]-pad_extent[0][1])/2
        else:
            ground_x_offset = dev_x_extent+5
        c.add(gd.Rectangle((i*(ground_x_offset-70), j*(idt_extent[1][1]+pad_tapery + v1_y)),
                           (i*(ground_x_offset), -j*(idt_extent[1][1]+pad_tapery + v1_y)),
                           layers['M2']))
        c.add(gd.Rectangle((-i*(pad_xoffset+pad_extent[0][1]),-j*(idt_extent[1][1]+pad_tapery + v1_y)),
                           (i*(ground_x_offset),-j*(idt_extent[1][1]+pad_tapery + v1_y+100)),
                           layers['M2']))

        # grounded idt taper connections
        c.add(gd.Polygon([(i*(pad_xoffset -n_idt * lmda/2), -j*(idt_extent[1][1]+ v1_y)),
                          (-i*(pad_xoffset+pad_extent[0][1]+70), -j*(idt_extent[1][1]+pad_tapery+ v1_y)),
                          (-i*(pad_xoffset+pad_extent[0][1]), -j*(idt_extent[1][1]+pad_tapery+ v1_y)),
                          (i*(pad_xoffset + n_idt*lmda/2), -j*(idt_extent[1][1]+ v1_y)),
                          ], layers['M2']))

        c.add(gd.Rectangle((i*(pad_xoffset-n_idt * lmda/2), -j*(idt_extent[1][1])),
                           (i*(pad_xoffset+n_idt * lmda/2), -j*(idt_extent[1][1] + v1_y)),
                           layers['M2']))
        c.add(gd.Rectangle((i*(pad_xoffset-n_idt * lmda/2), -j*(idt_extent[1][1])),
                           (i*(pad_xoffset+n_idt * lmda/2), -j*(idt_extent[1][1] + v1_y)),
                           layers['M1']))
    # bloat signal pad taper and subtract from ground routing to prevent shorting for small lengths with reflectors

    return c


def idt_cell(lib, layers, w_idt, s_idt, l_idt, n_idt, w_b, s_b, label='IDT'):
    cellname = label+f'_wIDT{w_idt}_sIDT{s_idt}_nIDT{n_idt:.1f}_lIDT{l_idt:.1f}_wb{w_b:.1f}_sb{s_b:.1f}'
    try:
        c = lib.new_cell(cellname)
    except ValueError:
        return lib.cells[cellname]
    for i in range(n_idt):
        xoff = i*2*(w_idt+s_idt)-n_idt*(w_idt+s_idt)
        c.add(gd.Rectangle((xoff, -l_idt/2), (xoff+w_idt, l_idt/2-s_b), layers['M1']))
        c.add(gd.Rectangle((xoff+w_idt+s_idt, -l_idt / 2+s_b), (xoff+2*w_idt+s_idt, l_idt/2), layers['M1']))
    # idt_extents = c.get_bounding_box()
    # c.add(gd.Rectangle((idt_extents[0][0], idt_extents[1][1]),
    #                    (idt_extents[1][0], idt_extents[1][1]+w_b),
    #                    layers['M1']))
    # c.add(gd.Rectangle((idt_extents[0][0], idt_extents[0][1]),
    #                    (idt_extents[1][0], idt_extents[0][1] - w_b),
    #                    layers['M1']))
    c.add(gd.Rectangle((-n_idt*(w_idt+s_idt), l_idt/2), (n_idt*(w_idt+s_idt)-s_idt, l_idt/2+w_b), layers['M1']))
    c.add(gd.Rectangle((-n_idt * (w_idt + s_idt), -l_idt / 2), (n_idt * (w_idt + s_idt)-s_idt, -l_idt / 2 - w_b), layers['M1']))
    return c


def idt_reflector(lib, layers, w_idt, s_idt, l_idt, n_idt, w_b):
    r = idt_cell(lib, layers, w_idt, s_idt, l_idt, n_idt, w_b, s_b=0, label='focIDT_reflector')
    return r


def focused_idt_cell(lib, layers, w_idt, s_idt, theta, g_idt, n_idt, w_b, s_b, label='focIDT', pad_contact=True):
    cellname = label+f'_wIDT{w_idt}_sIDT{s_idt}_nIDT{n_idt:.1f}_theta{theta:0.1f}_(g_idt{g_idt:0.1f}_wb{w_b:.1f}_sb{s_b:.1f}'
    ang = theta * np.pi / 360  # half angle, radians
    try:
        c = lib.new_cell(cellname)
    except ValueError:
        return lib.cells[cellname]

    for i in range(n_idt):
        xoff = i*2*(w_idt+s_idt)
        f1 = gd.Round((0, 0), g_idt/2+xoff+w_idt, g_idt/2+xoff, -ang, ang, layer=layers['M1'])
        sb1 = gd.Rectangle((s_b, 0), (g_idt/2+n_idt*2*(w_idt+s_idt), -s_b), layers['M1']).rotate(ang)
        c.add(gd.boolean(f1, sb1, 'not', layer=layers['M1']))

        f2 = gd.Round((0, 0), g_idt/2+xoff+2*w_idt+s_idt, g_idt/2+xoff+w_idt+s_idt, -ang, ang, layer=layers['M1'])
        sb2 = gd.Rectangle((s_b, 0), (g_idt / 2 + n_idt * 2 * (w_idt + s_idt),s_b), layers['M1']).rotate(-ang)
        c.add(gd.boolean(f2, sb2, 'not', layer=layers['M1']))
        # c.add(gd.Rectangle((xoff, -l_idt/2), (xoff+w_idt, l_idt/2-s_b), layers['M1']))
        # c.add(gd.Rectangle((xoff+w_idt+s_idt, -l_idt / 2+s_b), (xoff+2*w_idt+s_idt, l_idt/2), layers['M1']))
    outer_r = g_idt / 2 + n_idt * 2 * (w_idt + s_idt) - s_idt
    # c.add(gd.Rectangle((g_idt/2-outer_r*(1-np.cos(ang)), outer_r*np.sin(ang)),
    #                    (outer_r*np.cos(ang), outer_r*np.sin(ang)+w_b),
    #                    layers['M1']))
    # c.add(gd.Rectangle((g_idt / 2-outer_r*(1-np.cos(ang)), -outer_r*np.sin(ang)),
    #                    (outer_r*np.cos(ang), -(outer_r*np.sin(ang)+w_b)),
    #                     layers['M1']))
    c.add(gd.Rectangle((g_idt/2, 0), (outer_r, w_b), layer=layers['M1']).rotate(ang))
    c.add(gd.Rectangle((g_idt / 2, 0), (outer_r, -w_b), layer=layers['M1']).rotate(-ang))
    if pad_contact:
        c.add(gd.Round(center=(outer_r*np.cos(ang)-w_b*np.sin(ang), outer_r*np.sin(ang)+w_b*np.cos(ang)),
                       radius=outer_r-g_idt/2,
                       initial_angle=np.pi,
                       final_angle=np.pi + ang,
                       layer=layers['M1']))
        c.add(gd.Round(center=(outer_r*np.cos(ang)-w_b*np.sin(ang), -(outer_r*np.sin(ang)+w_b*np.cos(ang))),
                       radius=outer_r-g_idt/2,
                       initial_angle=np.pi,
                       final_angle=np.pi - ang,
                       layer=layers['M1']))
    # idt_extents = c.get_bounding_box()
    # c.add(gd.Rectangle((idt_extents[0][0], idt_extents[1][1]),
    #                    (idt_extents[1][0], idt_extents[1][1]+w_b),
    #                    layers['M1']))
    # c.add(gd.Rectangle((idt_extents[0][0], idt_extents[0][1]),
    #                    (idt_extents[1][0], idt_extents[0][1] - w_b),
    #                    layers['M1']))

    return c


def focused_idt_reflector(lib, layers, w_idt, s_idt, theta, g_idt, n_idt, w_b, label='focIDT_reflector'):
    return focused_idt_cell(lib, layers, w_idt, s_idt, theta, g_idt, n_idt, w_b, s_b=0, label=label, pad_contact=False)


def ewc_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b):
    #TODO: check phase for delay between two IDTs
    cellname = f'EWC_lamda{lmda:.1f}_bias{process_bias:.1f}_nIDT{n_idt:.1f}_lIDT{l_idt:.1f}_wb{w_b:.1f}_sb{s_b:.1f}'
    try:
        c = lib.new_cell(cellname)
    except ValueError:
        return lib.cells[cellname]
    for i in range(n_idt):
        xoff = i * lmda - n_idt * lmda/2
        c.add(gd.Rectangle((xoff+lmda * 4/16, -l_idt / 2), (xoff  + lmda * 6/16, l_idt / 2 - s_b), layers['M1']))
        c.add(gd.Rectangle((xoff, -l_idt / 2+ s_b), (xoff+ lmda/8, l_idt / 2), layers['M1']))
        c.add(gd.Rectangle((xoff + lmda * 9/16, -l_idt / 2 + s_b), (xoff + lmda * 13/16, l_idt / 2), layers['M1']))
    c.add(gd.Rectangle((-n_idt * lmda/2, l_idt / 2), (n_idt * lmda/2- 3*lmda/16, l_idt / 2 + w_b), layers['M1']))
    c.add(
        gd.Rectangle((-n_idt * lmda/2, -l_idt / 2), (n_idt * lmda/2 - 3*lmda/16, -l_idt / 2 - w_b), layers['M1']))
    # c.add(gd.Label('<-- fwd', (- n_idt * lmda / 2, l_idt / 2), anchor='se', layer=layers['M1']))
    return c


def dart_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b):
    #TODO: check phase for delay between two IDTs
    cellname = f'DART_lamda{lmda:.1f}_bias{process_bias:.1f}_nIDT{n_idt:.1f}_lIDT{l_idt:.1f}_wb{w_b:.1f}_sb{s_b:.1f}'
    try:
        c = lib.new_cell(cellname)
    except ValueError:
        return lib.cells[cellname]
    for i in range(n_idt):
        xoff = i * lmda - n_idt * lmda/2
        c.add(gd.Rectangle((xoff+lmda * 4/16, -l_idt / 2), (xoff + lmda * 6/16, l_idt / 2 - s_b), layers['M1']))
        c.add(gd.Rectangle((xoff, -l_idt / 2+ s_b), (xoff + lmda/8, l_idt / 2 ), layers['M1']))
        c.add(gd.Rectangle((xoff + lmda * 8/16, -l_idt / 2 + s_b), (xoff + lmda * 14/16, l_idt / 2), layers['M1']))
    c.add(gd.Rectangle((-n_idt * lmda/2, l_idt / 2), (n_idt * lmda/2- lmda/8, l_idt / 2 + w_b), layers['M1']))
    c.add(
        gd.Rectangle((-n_idt * lmda/2, -l_idt / 2), (n_idt * lmda/2- lmda/8, -l_idt / 2 - w_b), layers['M1']))
    #c.add(gd.Label('<-- fwd', (- n_idt * lmda/2,l_idt/2), anchor='se', layer=layers['M1']))
    return c


def split_finger_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b):
    #TODO: check phase for delay between two IDTs
    cellname = f'splitfinger_lamda{lmda:.1f}_bias{process_bias:.1f}_nIDT{n_idt:.1f}_lIDT{l_idt:.1f}_wb{w_b:.1f}_sb{s_b:.1f}'
    try:
        c = lib.new_cell(cellname)
    except ValueError:
        return lib.cells[cellname]
    for i in range(n_idt):
        xoff = i * lmda - n_idt * lmda/2
        c.add(gd.Rectangle((xoff, -l_idt / 2), (xoff + lmda/8, l_idt / 2 - s_b), layers['M1']))
        c.add(gd.Rectangle((xoff+lmda * 2/8, -l_idt / 2), (xoff + lmda * 3/8, l_idt / 2 - s_b), layers['M1']))
        c.add(gd.Rectangle((xoff + lmda * 4/8, -l_idt / 2 + s_b), (xoff + lmda * 5/8, l_idt / 2), layers['M1']))
        c.add(gd.Rectangle((xoff + lmda * 6 / 8, -l_idt / 2 + s_b), (xoff + lmda * 7 / 8, l_idt / 2), layers['M1']))
    c.add(gd.Rectangle((-n_idt * lmda/2, l_idt / 2), (n_idt * lmda/2- lmda/8, l_idt / 2 + w_b), layers['M1']))
    c.add(
        gd.Rectangle((-n_idt * lmda/2, -l_idt / 2), (n_idt * lmda/2- lmda/8, -l_idt / 2 - w_b), layers['M1']))
    return c



# WIP
def alignment_marks(lib, layers):
    l = 60
    t = 5

    c = lib.new_cell(f'Alignment_Mark_Layer{layer}')
    c.add(gd.Rectangle((-l -t/2), (-3*l/4, t/2)))
    c.add(gd.Rectangle((-3*l/4, -t), (-l / 4, t)))
    c.add(gd.Rectangle((-l / 4, -t / 2), (l / 4, t / 2)))
    c.add(gd.Rectangle((l / 4, -t), (3*l / 4, t)))
    c.add(gd.Rectangle((3*l / 4, - t / 2), (l, t / 2)))

lib = gd.GdsLibrary('Resonator Library')
gd.current_library = lib
ds = np.linspace(10, 100, 7)
ys = np.linspace(100, 200, 6)

layers = {
    'Trench': 0,
    'M1': 1,
    'M2': 2,
    }


pads = gsg_pad(lib, layers)
#bar_resonator_matrix(lib, pads, layers, ds, ys, trench=True)

test_idt = idt_device(lib, pads, layers, lmda=1, g_idt=100, idt_type=IDT_Type.STANDARD, l_idt=50,
                      reflector=True, g_r=.25, w_br=5, n_idtr=20, theta=60)
test_idt = idt_device(lib, pads, layers, lmda=1, g_idt=100, idt_type=IDT_Type.FOCUSED, l_idt=50,
                      reflector=True, g_r=.25, w_br=5, n_idtr=20, theta=60)
for i in IDT_Type:
    # test_idt = idt_device(lib, pads, layers, lmda=1, g_idt=100, idt_type=i, l_idt=50,
    #                       reflector=True, g_r=.25, w_br=5, n_idtr=20, theta=60)
    test_idt = idt_device(lib, pads, layers, lmda=1, g_idt=100, idt_type=i, l_idt=50,
                          reflector=False, g_r=.25, w_br=5, n_idtr=50, theta=60)
#focused_idt_cell(lib,layers,w_idt=.25,s_idt=.25,theta=60,g_idt=20,n_idt=20,w_b=10,s_b=3)
gd.LayoutViewer(lib)

#%%
lib.write_gds('tweed.gds')
