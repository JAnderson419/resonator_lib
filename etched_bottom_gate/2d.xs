

ar = 50000
csar62_t = 0.2
csar62_angle = 5 #TODO: fix interaction between angle and ar
ar_ang = Math.atan(Math.tan(csar62_angle*Math::PI/180)/ar)*180/Math::PI
depth(2.0*ar)
height(2.0*ar)
delta(0.001 * dbu)




lmarks = layer("11/0")
lidt = layer("1/0")
ilidt = layer("1/0").inverted
lpads = layer("2/0")
lfield = layer("3/0").inverted
lfox = layer("4/0").inverted
lgcontact = layer("4/0")
lte = layer("51/0")

si = bulk  # intrinsic
#isoal2o3 = deposit(0.025*ar) # ALD, maybe not needed?
gatemetal = deposit(0.2*ar)  # 200 nm Ni for EBL marks
gateox = deposit(0.023*ar)  # ALD Al2O3 or HfO2

te = mask(lte).grow(0.02*ar)  # solution deposited


#mask(lfield).etch(0.250*ar, :into => [gateox, gatemetal, te, si])

#mask(lgcontact).etch(0.1*ar, :into => gateox)

#fox = mask(lfox).grow(0.05*ar, 0.05*ar)# ALD or sputtered SiO2


# simple litho step
#idt = mask(lidt).grow(0.045*ar, 0.005*ar) # evaporated Ti/Al 

# full illustration of litho process
#PR = deposit(csar62_t*ar)
#PR_expose = mask(lidt).grow(csar62_t*ar, :bias => Math.tan(ar_ang*Math::PI/180)*csar62_t, :into => PR, :buried => csar62_t*ar, :taper => -ar_ang) # overhang profile
#etch(csar62_t*ar, :into => PR_expose)
#idt = grow(0.045*ar) # evaporated Ti/Al 
#mask(ilidt).etch(100*ar, :into => [idt, PR])

#padmet = mask(lpads).grow(0.3*ar, 0.02*ar) # evaporated Ti/Au pad 



output("300/0", si)
#output("302/0", isoal2o3)
output("301/0", gatemetal)
output("302/1", gateox)
output("303/0", te)
#output("304/0", idt)
#output("305/0", fox)
#output("306/0", padmet)
#output("307/0", PR)
