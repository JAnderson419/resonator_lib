
ar = 5 # aspect ratio

depth(2*ar)
height(5*ar)
delta(5 * dbu)


oxetch = layer("0/0")
m1 = layer("1/0")

sub = bulk
fox = deposit(0.5*ar)
rie = mask(oxetch).etch(0.55*ar, :into => fox)

metal = deposit(0.3*ar)
metal_liftoff = mask(m1.inverted).etch(0.3*ar, :into => metal)

output("300/0", sub)
output("301/0", fox)
output("302/0", metal)