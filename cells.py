# %%
import gdspy as gd
import numpy as np

def gsg_pad(lib, layers=[0,1], p=150, x=80, y=70):
    c = lib.new_cell(f'GSG_pad_P{p}_x{x}_y{y}')
    #  signal pad
    c.add(gd.Rectangle((-x/2, -y/2), (x/2, y/2), layer=layers['M2']))
    # rt = gd.Polygon([(x/2, -y/2),(x/2+40, -y/2+10),(x/2+40, y/2-10),(x/2, y/2)])

    #  ground pads
    c.add(gd.Rectangle((-x/2-20, -y/2-p), (x/2, y/2-p), layer=layers['M2']))
    c.add(gd.Rectangle((-x/2-20, -y/2+p), (x/2, y/2+p), layer=layers['M2']))

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


def idt_device(lib, pads, layers, lmda, g_idt, idt_type='standard', process_bias=0, n_idt=20, w_b=20, s_b=5, l_idt=None,
        reflector=False, g_r=None, w_br=None, n_idtr=None, w_r=None, s_r=None):

    # Parameter Setup
    if l_idt is None:
        l_idt = 100*lmda
    if idt_type == 'standard':
        cell_name = 'dIDT'
        w_idt = lmda / 4 + process_bias
        s_idt = lmda / 4 - process_bias
        idts = idt_cell(lib, layers, w_idt, s_idt, l_idt, n_idt, w_b, s_b)
    elif idt_type == 'EWC':
        cell_name = 'dEWC'
        idts = ewc_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
    elif idt_type == 'split':
        cell_name = 'dSplitFing'
        idts = split_finger_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
    elif idt_type == 'dart':
        cell_name = 'dDART'
        idts = dart_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b)
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
    idt_offset = g_idt/2 + idt_extent[1][0]
    c.add(gd.CellReference(idts, origin=(-idt_offset, 0), rotation=180)) # rotation puts idt finger at gap edge
    c.add(gd.CellReference(idts, origin=(idt_offset, 0), rotation=0))
    if reflector is True:
        r = idt_reflector(lib, layers, w_r, s_r, l_idt, n_idtr, w_br)
        r_extent = gd.CellReference(r).get_bounding_box()
        r_offset = g_idt/2 + (idt_extent[1][0]-idt_extent[0][0]) + g_r + r_extent[1][0]
        c.add(gd.CellReference(r, origin=(r_offset, 0)))
        c.add(gd.CellReference(r, origin=(-r_offset, 0), rotation=180))

    # generate pads
    pad_extent = pads.get_bounding_box()
    pad_tapery = 50  # offset distance for tapering from pad edge to IDT busbar
    v1_y = 20
    pad_yoffset = idt_extent[1][1]+pad_extent[1][0]+pad_tapery+ v1_y
    for (i, j) in [(-1, 1), (1, -1)]:
        c.add(gd.CellReference(pads,
                               rotation=i*90,
                               origin=(i*idt_offset, j*pad_yoffset)))
        c.add(gd.Polygon([(i*(idt_offset -n_idt * lmda/2), j*idt_extent[1][1]),
                          (i*(idt_offset-35), j*(idt_extent[1][1]+pad_tapery)),
                          (i*(idt_offset+35), j*(idt_extent[1][1]+pad_tapery)),
                          (i*(idt_offset + n_idt*lmda/2), j*idt_extent[1][1]),
                          ], layers['M1']))
        c.add(gd.Rectangle((i*(idt_offset-35), j*(idt_extent[1][1]+pad_tapery)),
                           (i*(idt_offset+35), j*(idt_extent[1][1]+pad_tapery + v1_y)),
                           layers['M2']))
        c.add(gd.Rectangle((i*(idt_offset-35), j*(idt_extent[1][1]+pad_tapery)),
                           (i*(idt_offset+35), j*(idt_extent[1][1]+pad_tapery + v1_y)),
                           layers['M1']))

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
    c.add(gd.Rectangle((-n_idt*(w_idt+s_idt), l_idt/2), (n_idt*(w_idt+s_idt), l_idt/2+w_b), layers['M1']))
    c.add(gd.Rectangle((-n_idt * (w_idt + s_idt), -l_idt / 2), (n_idt * (w_idt + s_idt), -l_idt / 2 - w_b), layers['M1']))
    return c

def focused_idt_cell(lib, layers, w_idt, s_idt, delay, theta, g_idt, n_idt, w_b, s_b, label='focIDT'):
    cellname = label+f'_wIDT{w_idt}_sIDT{s_idt}_nIDT{n_idt:.1f}delay{delay:.1f}_theta{theta:0.1f}(g_idt{g_idt:0.1f}_wb{w_b:.1f}_sb{s_b:.1f}'
    try:
        c = lib.new_cell(cellname)
    except ValueError:
        return lib.cells[cellname]
    c.add(gd.Round((0,0),delay/2+w_idt,delay/2,-theta*np.pi/360,theta*np.pi/360, layer=layers['M1']))

    for i in range(n_idt):
        xoff = delay/2 + i*2*(w_idt+s_idt)-n_idt*(w_idt+s_idt)
        c.add(gd.Round((0, 0), delay / 2 + xoff+w_idt, delay/2 + xoff, -theta * np.pi / 360, theta * np.pi / 360,
                       layer=layers['M1']))
        # c.add(gd.Rectangle((xoff, -l_idt/2), (xoff+w_idt, l_idt/2-s_b), layers['M1']))
        # c.add(gd.Rectangle((xoff+w_idt+s_idt, -l_idt / 2+s_b), (xoff+2*w_idt+s_idt, l_idt/2), layers['M1']))
    c.add(gd.Rectangle((delay/2, 0), (delay/2+n_idt*2*(w_idt+s_idt), w_b), layers['M1']).rotate(theta * np.pi / 360))
    c.add(gd.Rectangle((delay / 2, 0), (delay / 2 + n_idt * 2 * (w_idt + s_idt),-w_b), layers['M1']).rotate(-theta * np.pi / 360))
    # idt_extents = c.get_bounding_box()
    # c.add(gd.Rectangle((idt_extents[0][0], idt_extents[1][1]),
    #                    (idt_extents[1][0], idt_extents[1][1]+w_b),
    #                    layers['M1']))
    # c.add(gd.Rectangle((idt_extents[0][0], idt_extents[0][1]),
    #                    (idt_extents[1][0], idt_extents[0][1] - w_b),
    #                    layers['M1']))

    return c

def idt_reflector(lib, layers, w_idt, s_idt, l_idt, n_idt, w_b):
    r = idt_cell(lib, layers, w_idt, s_idt, l_idt, n_idt, w_b, s_b=0, label='IDT_reflector')
    return r

def ewc_cell(lib, layers, lmda, process_bias, l_idt, n_idt, w_b, s_b):
    #TODO: check phase for delay between two IDTs
    cellname = f'EWC_lamda{lmda:.1f}_bias{process_bias:.1f}_nIDT{n_idt:.1f}_lIDT{l_idt:.1f}_wb{w_b:.1f}_sb{s_b:.1f}'
    try:
        c = lib.new_cell(cellname)
    except ValueError:
        return lib.cells[cellname]
    for i in range(n_idt):
        xoff = i * lmda - n_idt * lmda/2
        c.add(gd.Rectangle((xoff, -l_idt / 2), (xoff + lmda/8, l_idt / 2 - s_b), layers['M1']))
        c.add(gd.Rectangle((xoff+lmda * 4/16, -l_idt / 2+ s_b), (xoff + lmda * 6/16, l_idt / 2 ), layers['M1']))
        c.add(gd.Rectangle((xoff + lmda * 9/16, -l_idt / 2 + s_b), (xoff + lmda * 13/16, l_idt / 2), layers['M1']))
    c.add(gd.Rectangle((-n_idt * lmda/2, l_idt / 2), (n_idt * lmda/2, l_idt / 2 + w_b), layers['M1']))
    c.add(
        gd.Rectangle((-n_idt * lmda/2, -l_idt / 2), (n_idt * lmda/2, -l_idt / 2 - w_b), layers['M1']))
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
        c.add(gd.Rectangle((xoff, -l_idt / 2), (xoff + lmda/8, l_idt / 2 - s_b), layers['M1']))
        c.add(gd.Rectangle((xoff+lmda * 4/16, -l_idt / 2+ s_b), (xoff + lmda * 6/16, l_idt / 2 ), layers['M1']))
        c.add(gd.Rectangle((xoff + lmda * 8/16, -l_idt / 2 + s_b), (xoff + lmda * 14/16, l_idt / 2), layers['M1']))
    c.add(gd.Rectangle((-n_idt * lmda/2, l_idt / 2), (n_idt * lmda/2, l_idt / 2 + w_b), layers['M1']))
    c.add(
        gd.Rectangle((-n_idt * lmda/2, -l_idt / 2), (n_idt * lmda/2, -l_idt / 2 - w_b), layers['M1']))
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
    c.add(gd.Rectangle((-n_idt * lmda/2, l_idt / 2), (n_idt * lmda/2, l_idt / 2 + w_b), layers['M1']))
    c.add(
        gd.Rectangle((-n_idt * lmda/2, -l_idt / 2), (n_idt * lmda/2, -l_idt / 2 - w_b), layers['M1']))
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
for i in ['standard', 'EWC', 'split', 'dart']:
    test_idt = idt_device(lib, pads, layers, 1, g_idt=3, idt_type=i, l_idt=100, reflector=True, g_r=10, n_idtr=50)
focused_idt_cell(lib,layers,.25,.25,20,60,.25,20,10,3)
gd.LayoutViewer(lib)

#%%
lib.write_gds('tweed.gds')