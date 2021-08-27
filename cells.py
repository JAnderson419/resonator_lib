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

class Bar_Type(Enum):
    BAR = 'bar'
    DISC = 'disc'



def gsg_pad(lib, layer, p=150, x=80, y=70, connect_grounds=False):
    c = lib.new_cell(f'GSG_pad_P{p}_x{x}_y{y}_tiedGnd{connect_grounds}')
    #  signal pad
    c.add(gd.Rectangle((-x/2, -y/2), (x/2, y/2), layer=layer))
    # rt = gd.Polygon([(x/2, -y/2),(x/2+40, -y/2+10),(x/2+40, y/2-10),(x/2, y/2)])

    #  ground pads
    c.add(gd.Rectangle((-x/2-20, -y/2-p), (x/2, y/2-p), layer=layer))
    c.add(gd.Rectangle((-x/2-20, -y/2+p), (x/2, y/2+p), layer=layer))

    if connect_grounds:
        c.add(gd.Rectangle((-x/2-40, -y/2-p), (-x/2-20, y/2+p), layer=layer))
    return c


def bar_resonator(lib, pads, layers, d=150, x=30, y=200, yoff=40, r=0, trench=False, name=None, type=Bar_Type.BAR):
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
    if name == None:
        name = type.value+f'_res_d{d}_x{x}_y{y}'
    if type == Bar_Type.DISC:
        name = name+f'_r{r}'
    c = lib.new_cell(name)

    pad_extent = gd.CellReference(pads).get_bounding_box()
    c.add(gd.CellReference(pads,
                           rotation=-90,
                           origin=(d / 2 + x/2, y / 2 + pad_extent[1][0]+30+yoff)))
    c.add(gd.CellReference(pads,
                           rotation=90,
                           origin=(-d / 2 - x / 2, -y / 2 - pad_extent[1][0]-30-yoff)))

    c.add(gd.Polygon([(pad_extent[0][1]+d/2+x/2, y/2+yoff+30),
                      (pad_extent[0][1]+d/2+x/2+70, y/2+yoff+30),
                      (pad_extent[0][1]-d/2-x/2+70, -y/2-yoff-30),
                      (pad_extent[0][1]-d/2-x/2, -y/2-yoff-30),
                      ], layers['M1']),)
    c.add(gd.Polygon([(-pad_extent[0][1]+d/2+x/2, y/2+yoff+30),
                      (-pad_extent[0][1]+d/2+x/2-70, y/2+yoff+30),
                      (-pad_extent[0][1]-d/2-x/2-70, -y/2-yoff-30),
                      (-pad_extent[0][1]-d/2-x/2, -y/2-yoff-30),
                      ], layers['M1']),)

    # Pad Tapers
    c.add(gd.Polygon([(d / 2, y / 2+yoff),
            (d / 2 + (x-70)/2, y / 2 + 30+yoff),
            (d / 2 + x - (x-70)/2, y / 2+30+yoff),
            (d / 2 + x, y / 2+yoff)
            ], layers['M1']),)
    c.add(gd.Polygon([(-d / 2, -y / 2-yoff),
            (-d / 2 - (x - 70) / 2, -y / 2 - 30-yoff),
            (-d / 2 - x + (x - 70) / 2, -y / 2 - 30-yoff),
            (-d / 2 - x, -y / 2-yoff)
            ], layers['M1']),)
    if type == Bar_Type.DISC:
        disc_cut = gd.Round((0, 0), radius=r, layer=layers['M1'])
    else:
        disc_cut = gd.Rectangle((0, 0), (0, 0), layer=layers['M1'])

    c.add(gd.boolean(gd.Rectangle((-d/2-x, -y/2-yoff), (-d/2, y/2), layers['M1']),
                     disc_cut,
                     'not', layer=layers['M1']))
    c.add(gd.boolean(gd.Rectangle((d / 2, -y / 2), (d / 2 + x, y / 2+yoff), layers['M1']),
                     disc_cut,
                     'not', layer=layers['M1']))

    if trench and d>10:
        c.add(gd.Rectangle((-d/2+5, -y/2-yoff/2), (d/2-5, y/2+yoff/2), layers['Trench']))
    extent = c.get_bounding_box()
    c.add(gd.Text(name, 20, (extent[0][0], extent[1][1]+5), layer=layers['M1']))
    return c


def bar_resonator_matrix(lib, pads, layers, ds, xs, ys, rs, trench=False, name='Resonator Matrix', **kwargs):

    m = lib.new_cell(name)
    max_x = 0
    max_y = 0
    cells = []
    x0 = []
    y0 = []

    # handles parsing of d,x,y,r to find two arrays to iterate over

    pd, p2, p1i, p2i = [None]*4
    p1 = np.empty(0)
    p2 = np.empty(0)

    ps = [None]*4
    for i, p in enumerate([ds, xs, ys, rs]):
        p = np.asarray(p)
        if p.size == 1:
            ps[i] = p
        elif(p1.size == 0):
            p1 = p
            p1i = i
        elif(p2.size == 0):
            p2 = p
            p2i = i
        else:
            raise ValueError("More than two series given.")

    for i, ps[p1i] in enumerate(p1):
        for j, ps[p2i] in enumerate(p2):
            c = bar_resonator(lib, pads, layers, d=ps[0], x=ps[1], y=ps[2], r=ps[3], trench=trench, **kwargs)
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
               process_bias: float=0, n_idt: int=20, w_b: float=5, s_b: float=1, l_idt: float=None,
               reflector: bool=False, g_r: float=None, w_br: float=None, n_idtr: int=None, w_r: float=None,
               s_r: float=None, theta: float=None, cell_prefix='') -> gd.Cell:

    pad_tapery = 50  # offset distance for tapering from pad edge to IDT busbar
    v1_y = 10
    M2_OVERLAP = 1
    dev_x_extent = 0
    cell_name = cell_prefix
    # Parameter Setup
    if l_idt is None:
        l_idt = 100*lmda
    if idt_type == IDT_Type.STANDARD:
        cell_name = cell_name+'dIDT'
        w_idt = lmda / 4 + process_bias
        s_idt = lmda / 4 - process_bias
        idts = idt_cell(lib, layers, w_idt, s_idt, l_idt, n_idt, w_b, s_b)
        bb_offset = s_idt
    elif idt_type == IDT_Type.EWC:
        if process_bias != 0:
            raise NotImplementedError("Process bias not implemented for SPUDT devices")
        cell_name = cell_name+'dEWC'
        idts = ewc_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
        bb_offset = 3*lmda/16
    elif idt_type == IDT_Type.SPLIT_FINGER:
        cell_name = cell_name+'dSplitFing'
        idts = split_finger_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
        bb_offset = lmda/8
    elif idt_type == IDT_Type.DART:
        if process_bias != 0:
            raise NotImplementedError("Process bias not implemented for SPUDT devices")
        cell_name = cell_name+'dDART'
        idts = dart_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
        bb_offset = lmda/8
    elif idt_type == IDT_Type.FOCUSED:
        if theta is None:
            raise ValueError("No angle defined for focused IDT.")
        else:
            cell_name = cell_name+f'dFoc_thta{theta:.1f}'
            w_idt = lmda / 4 + process_bias
            s_idt = lmda / 4 - process_bias
            idts = focused_idt_cell(lib, layers, w_idt, s_idt, theta, g_idt, n_idt, w_b, s_b)
            bb_offset = s_idt
    text_label = cell_name + f'_lmbda{lmda:.1f}_g{g_idt:.1f}_nI{n_idt:.1f}_lI{l_idt:.1f}'
    cell_name = cell_name+f'_lambda{lmda:.1f}_g{g_idt:.1f}_nIDT{n_idt:.1f}_IDTl{l_idt:.1f}_wb{w_b:.1f}_sb{s_b:.1f}_bias{process_bias:.1f}'
    if reflector is True:
        if g_r is None: g_r = 5*lmda # TODO: Change default value based on simulated performance
        if w_br is None: w_br = w_b
        if w_r is None: w_r = lmda / 4 + process_bias
        if s_r is None: s_r = lmda / 4 - process_bias
        if n_idtr is None: n_idtr = n_idt
        cell_name = cell_name+f'_gr{g_r:.1f}_wr{w_r:.1f}_sr{s_r:.1f}_wbr{w_br:.1f}_nR{n_idtr:.1f}'
        text_label = text_label+f'_gr{g_r:.1f}_nR{n_idtr:.1f}'
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
            r = focused_idt_reflector(lib, layers, w_r, s_r, theta, g_idt+2*(g_r+n_idt*lmda-lmda/4), n_idtr, w_br)
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


    # JOEL field rectangle to prevent stitching errors in IDT
    cell_idt_extents = c.get_bounding_box()
    if cell_idt_extents[1][1]-cell_idt_extents[1][0] + 2*(w_b+v1_y) < 980 and cell_idt_extents[0][1]-cell_idt_extents[0][0] < 980:
        c.add(gd.Rectangle([cell_idt_extents[0][0]-5, cell_idt_extents[0][1]-5],
                           [cell_idt_extents[1][0]+5, cell_idt_extents[1][1]+5],
                           layers['JOEL_FIELD']))
    elif idt_extent[1][1]-idt_extent[1][0] < 980 and idt_extent[0][1]-idt_extent[0][0] < 980:
        c.add(gd.Rectangle([idt_extent[0][0]-5-idt_offset, idt_extent[0][1]-5 - (w_b+v1_y)],
                           [idt_extent[1][0]+5-idt_offset, idt_extent[1][1]+5 + (w_b+v1_y)],
                           layers['JOEL_FIELD']))
        c.add(gd.Rectangle([idt_extent[0][0]-5+idt_offset, idt_extent[0][1]-5 - (w_b+v1_y)],
                           [idt_extent[1][0]+5+idt_offset, idt_extent[1][1]+5 + (w_b+v1_y)],
                           layers['JOEL_FIELD']))
    else:
        pass  # can't break IDT into one field.

    # generate pads
    pad_extent = pads.get_bounding_box()

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
        c.add(gd.Polygon([(i*(pad_xoffset -n_idt * lmda/2- M2_OVERLAP), j*(idt_extent[1][1] + v1_y)),
                          (i*(pad_xoffset-35), j*(idt_extent[1][1] + v1_y+pad_tapery)),
                          (i*(pad_xoffset+35), j*(idt_extent[1][1] + v1_y+pad_tapery)),
                          (i*(pad_xoffset + n_idt*lmda/2 - bb_offset+M2_OVERLAP), j*(idt_extent[1][1] + v1_y)),
                          ], layers['M2']))
        c.add(gd.Rectangle((i*(pad_xoffset-n_idt * lmda/2-M2_OVERLAP), j*(idt_extent[1][1])),
                           (i*(pad_xoffset+n_idt * lmda/2 - bb_offset+M2_OVERLAP), j*(idt_extent[1][1] + v1_y)),
                           layers['M2']))
        c.add(gd.Rectangle((i*(pad_xoffset-n_idt * lmda/2), j*(idt_extent[1][1])),
                           (i*(pad_xoffset+n_idt * lmda/2 - bb_offset), j*(idt_extent[1][1] + v1_y)),
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
        c.add(gd.Polygon([(i*(pad_xoffset -n_idt * lmda/2 - M2_OVERLAP), -j*(idt_extent[1][1]+ v1_y)),
                          (-i*(pad_xoffset+pad_extent[0][1]+70), -j*(idt_extent[1][1]+pad_tapery+ v1_y)),
                          (-i*(pad_xoffset+pad_extent[0][1]), -j*(idt_extent[1][1]+pad_tapery+ v1_y)),
                          (i*(pad_xoffset + n_idt*lmda/2 - bb_offset + M2_OVERLAP), -j*(idt_extent[1][1]+ v1_y)),
                          ], layers['M2']))

        c.add(gd.Rectangle((i*(pad_xoffset-n_idt * lmda/2 - M2_OVERLAP), -j*(idt_extent[1][1])),
                           (i*(pad_xoffset+n_idt * lmda/2 - bb_offset + M2_OVERLAP), -j*(idt_extent[1][1] + v1_y)),
                           layers['M2']))
        c.add(gd.Rectangle((i*(pad_xoffset-n_idt * lmda/2), -j*(idt_extent[1][1])),
                           (i*(pad_xoffset+n_idt * lmda/2 - bb_offset), -j*(idt_extent[1][1] + v1_y)),
                           layers['M1']))
    # TODO: bloat signal pad taper and subtract from ground routing to prevent shorting for small lengths with reflectors?

    # Cell label in M2
    c.add(gd.Text(text_label,
                  9,
                  (-pad_xoffset-(pad_extent[1][1]-pad_extent[0][1])/2,
                    pad_yoffset-pad_extent[0][0]+5),
                  layer=layers['M2']))

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
ds = [1, 1.5, 2, 2.5, 3, 5, 7, 10]
ys = [20, 30, 40, 50]
rs = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

layers = {
    'Trench': 0,
    'M1': 1,
    'M2': 2,
    'JOEL_FIELD': 50
    }


pads = gsg_pad(lib, layers['M1'])
pads2 = gsg_pad(lib, layers['M2'], connect_grounds=True)
bar_resonator_matrix(lib, pads, layers, ds=ds, xs=30, ys=ys, rs=0, trench=False, yoff=10, name='matrix1')
bar_resonator_matrix(lib, pads, layers, rs=rs, ds=[1,2], xs=30, ys=10, yoff=4, type=Bar_Type.DISC, name='matrix2')
lmda1 = 1.5
g_idt1 = 100*lmda1
l_idt1 = 50*lmda1
g_r1 = lmda1/4

prefixes = ['','g']
# for j,p in enumerate([pads,pads2]):
#
#     idt_device(lib, p, layers, lmda=lmda1, g_idt=g_idt1, idt_type=IDT_Type.STANDARD, l_idt=l_idt1, s_b=lmda1,
#                           reflector=True, g_r=g_r1, w_br=3, n_idtr=25, theta=60, cell_prefix=prefixes[j])
#     idt_device(lib, p, layers, lmda=lmda1, g_idt=g_idt1, idt_type=IDT_Type.FOCUSED, l_idt=l_idt1, s_b=lmda1,
#                           reflector=True, g_r=g_r1, w_br=3, n_idtr=25, theta=60, cell_prefix=prefixes[j])
#     for i in IDT_Type:
#         # test_idt = idt_device(lib, pads, layers, lmda=1, g_idt=100, idt_type=i, l_idt=50,
#         #                       reflector=True, g_r=.25, w_br=5, n_idtr=20, theta=60)
#         idt_device(lib, p, layers, lmda=lmda1, g_idt=g_idt1, idt_type=i, l_idt=l_idt1, s_b=lmda1,
#                               reflector=False, g_r=g_r1, w_br=3, n_idtr=50, theta=60, cell_prefix=prefixes[j])
#     #focused_idt_cell(lib,layers,w_idt=.25,s_idt=.25,theta=60,g_idt=20,n_idt=20,w_b=10,s_b=3)
gd.LayoutViewer(lib)

# %%
lib.write_gds('test.gds')
