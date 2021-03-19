# %%
import gdspy as gd
import numpy as np



def gsg_pad(lib, layers=[0,1], p=150, x=80, y=70):
    c = lib.new_cell(f'GSG_pad_P{p}_x{x}_y{y}')
    #  signal pad
    c.add(gd.Rectangle((-x/2, -y/2), (x/2, y/2), layer=layers[1]))
    # rt = gd.Polygon([(x/2, -y/2),(x/2+40, -y/2+10),(x/2+40, y/2-10),(x/2, y/2)])

    #  ground pads
    c.add(gd.Rectangle((-x/2-20, -y/2-p), (x/2, y/2-p), layer=layers[1]))
    c.add(gd.Rectangle((-x/2-20, -y/2+p), (x/2, y/2+p), layer=layers[1]))

    c.add(gd.Rectangle((-x/2-40, -y/2-p), (-x/2-20, y/2+p), layer=layers[1]))
    return c

def bar_resonator(lib, pads, layers=[0, 1], d=150, x=30, y=200, yoff=40, trench=False):
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
    c.add(gd.Rectangle((-d/2-x, -y/2-yoff), (-d/2, y/2), layers[1]))
    c.add(gd.Rectangle((d / 2, -y / 2), (d / 2 + x, y / 2+yoff), layers[1]))
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
                      ], layers[1]))


    if trench and d>10:
        c.add(gd.Rectangle((-d/2+5, -y/2-yoff/2), (d/2-5, y/2+yoff/2), layers[0]))
    extent = c.get_bounding_box()
    c.add(gd.Text(f'bar_res_d{d}_x{x}_y{y}', 20, (extent[0][0], extent[1][1]+5), layers[1]))
    return c



def bar_resonator_matrix(lib, pads, ds, ys, trench=False):
    m = lib.new_cell(f'Resonator Matrix')
    max_x = 0
    max_y = 0
    cells = []
    x0 = []
    y0 = []
    for i, d in enumerate(ds):
        for j, y in enumerate(ys):
            c = bar_resonator(lib, pads, d=d, y=y, trench=trench)
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
pads = gsg_pad(lib)

bar_resonator_matrix(lib, pads, ds, ys, trench=True)
gd.LayoutViewer(lib)

#%%
lib.write_gds('tweed.gds')