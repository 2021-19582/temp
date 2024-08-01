import numpy
from pyscf import gto,scf
import sys

#==================================================================
# MOLECULE
#==================================================================
mol = gto.Mole()
mol.verbose = 5
mol.max_memory = 12500

#==================================================================
# Coordinates and basis
#==================================================================
#model = '2feIII_lunocloc'
model = 'feIIIfeII_lunocloc'

if '2feIII_' in model:
    atom = """
 Fe                 5.22000000    1.05000000   -7.95000000
 S                  3.86000000   -0.28000000   -9.06000000
 S                  5.00000000    0.95000000   -5.66000000
 S                  4.77000000    3.18000000   -8.74000000
 S                  7.23000000    0.28000000   -8.38000000
 Fe                 5.88000000   -1.05000000   -9.49000000
 S                  6.10000000   -0.95000000  -11.79000000
 S                  6.33000000   -3.18000000   -8.71000000
 C                  6.00000000    4.34000000   -8.17000000
 H                  6.46000000    4.81000000   -9.01000000
 H                  5.53000000    5.08000000   -7.55000000
 H                  6.74000000    3.82000000   -7.60000000
 C                  3.33000000    1.31000000   -5.18000000
 H                  2.71000000    0.46000000   -5.37000000
 H                  3.30000000    1.54000000   -4.13000000
 H                  2.97000000    2.15000000   -5.73000000
 C                  5.10000000   -4.34000000   -9.28000000
 H                  5.56000000   -5.05000000   -9.93000000
 H                  4.67000000   -4.84000000   -8.44000000
 H                  4.34000000   -3.81000000   -9.81000000
 C                  7.77000000   -1.31000000  -12.27000000
 H                  7.84000000   -1.35000000  -13.34000000
 H                  8.42000000   -0.54000000  -11.90000000
 H                  8.06000000   -2.25000000  -11.86000000
"""
    charge = -2
    twos = 10 
elif 'feIIIfeII_' in model:
    atom = """
Fe	5.48	1.15	-8.03
S	4.05	-0.61	-8.75
S	5.47	1.25	-5.58
S	4.63	3.28	-8.77
S	7.49	0.42	-9.04
Fe	6.04	-1.22	-9.63
S	5.75	-1.5	-12.05
S	6.86	-3.41	-8.86
C	5.51	4.45	-7.51
H	6.49	4.83	-7.92
H	4.87	5.33	-7.25
H	5.72	3.84	-6.59
C	3.6	1.7	-5.54
H	3.01	0.8	-5.82
H	3.28	2.06	-4.52
H	3.42	2.48	-6.31
C	5.21	-4.22	-9.46
H	5.1	-4.01	-10.55
H	5.21	-5.32	-9.26
H	4.37	-3.72	-8.93
C	7.63	-1.85	-12.24
H	7.9	-2.06	-13.31
H	8.2	-0.96	-11.86
H	7.89	-2.72	-11.59
"""
    charge = -3
    twos = 9 
else:
    assert False 

ano_fe = gto.basis.parse('''
Fe    S
4316265.                     0.00015003            -0.00004622             0.00001710            -0.00000353             0.00000423
 646342.4                    0.00043597            -0.00013445             0.00004975            -0.00001026             0.00001234
 147089.7                    0.00120365            -0.00037184             0.00013758            -0.00002838             0.00003400
  41661.52                   0.00312635            -0.00096889             0.00035879            -0.00007397             0.00008934
  13590.77                   0.00814591            -0.00253948             0.00094021            -0.00019410             0.00023098
   4905.750                  0.02133892            -0.00673001             0.00249860            -0.00051496             0.00062709
   1912.746                  0.05470838            -0.01768160             0.00657103            -0.00135801             0.00160135
    792.6043                 0.12845394            -0.04375410             0.01640473            -0.00338297             0.00416181
    344.8065                 0.25203824            -0.09601111             0.03637157            -0.00754121             0.00877359
    155.8999                 0.35484986            -0.16998599             0.06664937            -0.01380066             0.01738346
     72.23091                0.27043078            -0.18456376             0.07553682            -0.01588736             0.01718943
     32.72506                0.06476086             0.05826300            -0.02586806             0.00570363            -0.00196602
     15.66762               -0.00110466             0.52163758            -0.31230230             0.06807261            -0.09285258
      7.503483               0.00184555             0.49331062            -0.44997654             0.10526256            -0.11350600
      3.312223              -0.00085600             0.08632670             0.14773374            -0.04562463             0.01812457
      1.558471               0.00037119            -0.00285017             0.72995709            -0.21341607             0.41268036
      0.683914              -0.00014687             0.00165569             0.38458847            -0.24353659             0.10339104
      0.146757               0.00006097            -0.00049176             0.01582890             0.34358715            -0.89083095
      0.070583              -0.00005789             0.00047608            -0.00949537             0.46401833            -0.80961283
      0.031449               0.00002770            -0.00022820             0.00308038             0.34688312             1.52308946
      0.012580              -0.00000722             0.00006297            -0.00100526             0.01225841             0.09142619
Fe    P
   7721.489                  0.00035287            -0.00012803            -0.00013663             0.00003845
   1829.126                  0.00196928            -0.00071517            -0.00077790             0.00021618
    593.6280                 0.00961737            -0.00352108            -0.00375042             0.00105697
    226.2054                 0.03724273            -0.01379065            -0.01516741             0.00418424
     95.26145                0.11332297            -0.04331452            -0.04705206             0.01307817
     42.85920                0.25335172            -0.10061222            -0.11529630             0.03095510
     20.04971                0.38104215            -0.16161377            -0.17017078             0.04896849
      9.620885               0.30703250            -0.11214083            -0.13220830             0.03516849
      4.541371               0.08654534             0.18501865             0.53797582            -0.08338612
      2.113500               0.00359924             0.47893080             0.61199701            -0.17709305
      0.947201               0.00144059             0.40514792            -0.64465308            -0.11907766
      0.391243              -0.00029901             0.09872160            -0.61225551             0.12237413
      0.156497               0.00020351            -0.00148592             0.10798966             0.54998130
      0.062599              -0.00009626             0.00222977             0.37358045             0.39970337
      0.025040               0.00002881            -0.00072259             0.18782870             0.08298275
Fe    D
    217.3688                 0.00096699            -0.00098327
     64.99976                0.00793294            -0.00789694
     24.77314                0.03548314            -0.03644790
     10.43614                0.10769519            -0.10760712
      4.679653               0.22555488            -0.26104796
      2.125622               0.31942979            -0.29085509
      0.945242               0.32354390             0.01254821
      0.402685               0.24338270             0.40386046
      0.156651               0.10680569             0.38672483
      0.062660               0.02052711             0.24394500
Fe    F
     11.2749                 0.03802196
      4.4690                 0.25501829
      1.7713                 0.50897998
       .7021                 0.35473516
       .2783                 0.12763297
       .1103                 0.01946831
''')
ano_s = gto.basis.parse('''
S    S
 346348.23                   0.00029092            -0.00008101             0.00002280            -0.00002342
  49391.146                  0.00094665            -0.00026388             0.00007421            -0.00007637
  14610.990                  0.00211923            -0.00059263             0.00016719            -0.00017089
   5187.2095                 0.00602161            -0.00168762             0.00047372            -0.00048999
   1980.9676                 0.01659726            -0.00470357             0.00133106            -0.00135293
    784.63139                0.04683989            -0.01350607             0.00379801            -0.00392820
    317.32779                0.12242581            -0.03711959             0.01058108            -0.01067637
    130.01976                0.27022435            -0.09114723             0.02601970            -0.02661813
     53.738208               0.40541657            -0.17780418             0.05279001            -0.05183027
     22.345896               0.25571527            -0.17069592             0.05191878            -0.05362245
      9.3332512              0.02671693             0.17423163            -0.05692828             0.07080611
      3.9111868              0.00390332             0.63283682            -0.29095190             0.28624981
      1.6432066             -0.00204189             0.35886635            -0.31632828             0.49626307
       .69174560             0.00149034            -0.00096444             0.29309938            -1.07109282
       .29167360            -0.00062107             0.00659942             0.66787427            -0.79207470
       .12314410             0.00044167            -0.00522266             0.28725199             1.19881853
       .04925760            -0.00000324             0.00029385             0.01516887             0.32475750
S    P
   1129.1269                 0.00090586            -0.00022454             0.00021481
    274.03515                0.00600033            -0.00148850             0.00137868
     97.402584               0.02531166            -0.00634942             0.00612706
     38.085518               0.09198941            -0.02344282             0.02170457
     15.471033               0.24165028            -0.06370257             0.06243734
      6.4056590              0.41029096            -0.11197083             0.10108619
      2.6812828              0.35219287            -0.10347278             0.11708231
      1.1300050              0.08281645             0.10903888            -0.16261828
       .47839950             0.00009280             0.41838741            -0.66461283
       .20318050            -0.00223897             0.43694091            -0.00256719
       .08649230            -0.00131050             0.18619701             0.71028173
       .03459690            -0.00022092             0.02837119             0.32134414
S    D
      3.0053679              0.03581500
      1.2172976              0.16734186
       .49305560             0.57605251
       .19970780             0.35014066
       .07988310             0.04467461
''')
ano_c = gto.basis.parse('''
C    S
  50557.501                  0.0001128874          -0.0000250742           0.0000161788
   7524.7856                 0.0005295373          -0.0001175945           0.0000749712
   1694.3276                 0.0024500383          -0.0005456275           0.0003573467
    472.82279                0.0100539847          -0.0022450771           0.0014349316
    151.71075                0.0354539806          -0.0080434718           0.0055086779
     53.918746               0.1044071086          -0.0244258871           0.0169149690
     20.659311               0.2412894918          -0.0613679118           0.0514166524
      8.3839760              0.3834225483          -0.1177767835           0.1137343741
      3.5770150              0.3078514393          -0.1550487783           0.2337183944
      1.5471180              0.0687244299          -0.0193331686          -0.0718378257
       .61301300             0.0002224445           0.3996930678          -1.0490908711
       .24606800             0.0019767489           0.5589420949          -0.0666273399
       .09908700             0.0020578594           0.1711195023           1.0119993798
       .03468000             0.0004182020           0.0074562446           0.1656046386
C    P
     83.333155               0.0013446560          -0.0017715885
     19.557611               0.0102355550          -0.0145621810
      6.0803650              0.0452006710          -0.0574573357
      2.1793170              0.1410756198          -0.2126582848
       .86515000             0.3047388085          -0.5271712924
       .36194400             0.3995281011          -0.0929697730
       .15474000             0.2719163432           0.6630031669
       .06542900             0.0585766933           0.3360877793
       .02290000            -0.0001263813           0.0054695244
C    D
      1.9000000              0.1424518048
       .66500000             0.5591973019
       .23275000             0.4430004380
       .08146300             0.0410665370
''')
ano_h = gto.basis.parse('''
H    S
    188.61445                 .00096385             -.0013119
     28.276596                .00749196             -.0103451
      6.4248300               .03759541             -.0504953
      1.8150410               .14339498             -.2073855
       .59106300              .34863630             -.4350885
       .21214900              .43829736             -.0247297
       .07989100              .16510661              .32252599
       .02796200              .02102287              .70727538
H    P
      2.3050000               .11279019
       .80675000              .41850753
       .28236200              .47000773
       .09882700              .18262603
''')

mpg = 'c1'  # point group
mol = gto.M(atom=atom, symmetry=mpg, basis= {'Fe': ano_fe, 'S': ano_s, 'C': ano_c, 'H': ano_h},
            spin=twos, charge=charge, verbose=5)
model = 'feIIIfeII_13o17eR_lunocloc'
method= 'casci'
 
#==================================================================
# UKS MF 
#==================================================================
if '_rhf' in model or '_rloc' in model or '_dft' in model or '_lunoloc' in model or '_lunocloc' in model:
    #self.mf = scf.ROHF(self.mol).x2c()
    mf = scf.ROHF(mol)
elif'_uhf' in model:
    #self.mf = scf.UHF(self.mol).x2c()
    mf = scf.UHF(mol)

# #==================================================================
# # Localization of nat orbs of spin averaged UKS 
# #==================================================================
# from pyxray.utils.lunoloc import dumpLUNO
# filename = '%s.h5' % model
# lmo, enorb, occ = dumpLUNO(mol, mf.mo_coeff, mf.mo_energy, thresh=0.05, dumpname=filename)

# with open('lunoloc.molden','w') as thefile:
#     molden.header(mol, thefile)
#     molden.orbital_coeff(mol, thefile, lmo, ene=enorb, occ=occ)
# sys.exit()


lib_pyxray_path='/home/amychoi7/git/pyxray2/pyxray'
data_path = '%s/data/fe2s2' % (lib_pyxray_path)
if '_lunocloc' in model:
    if '2feIII_' in model:
        chkfile = "%s/2feIII_lunocloc.h5" % (data_path)
    elif 'feIIIfeII_' in model:
        chkfile = "%s/feIIIfeII_lunocloc.h5" % (data_path)
    else:
        assert False
else:
    assert False

if chkfile is None:
    self.mf.kernel()
    self.mo_coeff = self.mf.mo_coeff
    self.mo_occ = self.mf.mo_occ
    self.mo_energy = self.mf.mo_energy
    if MPI is not None:
        from mpi4py import MPI as MPIPY
        comm = MPIPY.COMM_WORLD
        self.mo_occ = comm.bcast(self.mo_occ, root=0)
        self.mf.mo_coeff = comm.bcast(self.mf.mo_coeff, root=0)
        self.mf.mo_energy = comm.bcast(self.mf.mo_energy, root=0)
else:
    # load h5py
    if '.h5' == chkfile[-3:] and ('_rhf' in model or '_rloc' in model or '_dft' in model):
        f = h5py.File(chkfile,'r')
        self.mo_coeff = np.array(f['scf']['mo_coeff'])
        self.mo_occ = np.array(f['scf']['mo_occ'])
        self.mo_energy = np.array(f['scf']['mo_energy'])
        f.close()
        self.mf.mo_occ = self.mo_occ
        self.mf.mo_energy = self.mo_energy
    elif '.h5' == chkfile[-3:] and ('_lunoloc' in model or '_lunocloc' in model):
        f = h5py.File(chkfile,'r')
        self.mo_coeff = np.array(f['luno']['mo_coeff'])
        #self.mo_occ = np.array(f['luno']['mo_occ'])
        #self.mo_energy = np.array(f['luno']['mo_energy'])
        f.close()
        #self.mf.mo_occ = self.mo_occ
        #self.mf.mo_energy = self.mo_energy
    # load numpy
    elif '.npy' in chkfile[-4:]:
        self.mo_coeff = np.load(chkfile)
    else:
        assert False
    self.mf.mo_coeff = self.mo_coeff
#==================================================================
# Further localization for atomistic core 2p orbital of Fe 
#                      for active orbitals (for DMRG)
#==================================================================
# active space index for LUNO 

####################
# define CAS model #
####################
if '2feIII_' in model:
    if n_pcore = 6
        elif '19o28eL' in model:
            act0 = [13,14,16] # feL 2p
            act1 = [] # feR 2p
            act2 = [71,72,73,74,77,78,93,94,95,96] # fe 3d
            act3 = [83,84,85,86,87,88] # s 3p
            idx = act0+act1+act2+act3
            na = len(idx) - 5
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act3
            n_pcore = 3
        elif '19o28eR' in model:
            act0 = [] # feL 2p
            act1 = [11,12,15] # feR 2p
            act2 = [71,72,73,74,77,78,93,94,95,96] # fe 3d
            act3 = [83,84,85,86,87,88] # s 3p
            idx = act0+act1+act2+act3
            na = len(idx) - 5
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act3
            n_pcore = 3
        elif 'gs' in model:
            act0 = [] # feL 2p
            act1 = [] # feR 2p
            act2 = [71,72,73,74,77,78,93,94,95,96] # fe 3d
            act3 = [83,84,85,86,87,88] # s 3p
            idx = act0+act1+act2+act3
            na = len(idx) - 5
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act3
            n_pcore = 0
        else:
            assert False
    elif '_rloc' in model:
        if '22o34e' in model:
            act0 = [77,78,79] # feL 2p
            act1 = [80,81,82] # feR 2p
            act2 = [83,84,85,86,87] # feL 3d
            act3 = [88,89,90,91,92,93] # Sc 3p
            act4 = [94,95,96,97,98] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 6
        else:
            assert False
    elif '_lunoloc' in model:
        if '22o34e' in model:
            act0 = [12,15,16] # feL 2p
            act1 = [11,13,14] # feR 2p
            act2 = [90,92,95,96,98] # feL 3d
            act3 = [77,80,83,84,81,76] # Sc 3p
            act4 = [89,91,93,94,97] # feR 3d

            sp_sigma_terminal = [75,79,78,82] # feL 3d
            sp_pi_terminal = [85,86,87,88] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 6
        elif '13o16eL' in model:
            act0 = [12,15,16] # feL 2p
            act2 = [90,92,95,96,98] # feL 3d
            act4 = [89,91,93,94,97] # feR 3d

            idx = act0+act2+act4
            na = len(idx) - 5
            nb = len(idx) - 5
            act_2p = act0
            act_3d = act2+act4
            n_pcore = 3
        elif '13o16eR' in model:
            act1 = [11,13,14] # feR 2p
            act2 = [90,92,95,96,98] # feL 3d
            act4 = [89,91,93,94,97] # feR 3d

            idx = act1+act2+act4
            na = len(idx) - 5
            nb = len(idx) - 5
            act_2p = act1
            act_3d = act2+act4
            n_pcore = 3
        else:
            assert False
    elif '_lunocloc' in model:
        if '22o34e' in model:
            act0 = [14,16,17] # feL 2p
            act1 = [9,10,15] # feR 2p
            act2 = [91,92,93,96,97] # feL 3d
            act3 = [83,84,85,86,87,88] # Sc 3p
            act4 = [89,90,94,95,98] # feR 3d

            sp_sigma_terminal = [79,80,81,82] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 6
        else:
            assert False
    else:
        assert False
elif 'feIIIfeII_' in model:
    if '_rloc' in model:
        if '22o35e' in model:
            act0 = [78,79,82] # feL 2p
            act1 = [77,80,81] # feR 2p
            act2 = [83,84,85,86,87] # feL 3d
            act3 = [88,89,90,91,92,93] # Sc 3p
            act4 = [94,95,96,97,98] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 4
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 6
        elif '19o29eL' in model:
            act0 = [78,79,82] # feL 2p
            act1 = [] # feR 2p
            act2 = [83,84,85,86,87] # feL 3d
            act3 = [88,89,90,91,92,93] # Sc 3p
            act4 = [94,95,96,97,98] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 4
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 3
        elif '19o29eR' in model:
            act0 = [] # feL 2p
            act1 = [77,80,81] # feR 2p
            act2 = [83,84,85,86,87] # feL 3d
            act3 = [88,89,90,91,92,93] # Sc 3p
            act4 = [94,95,96,97,98] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 4
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 3
        elif '13o17eL' in model:
            act0 = [78,79,82] # feL 2p
            act1 = [] # feR 2p
            act2 = [83,84,85,86,87] # feL 3d
            act3 = [] # Sc 3p
            act4 = [94,95,96,97,98] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 4
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 3
        elif '13o17eR' in model:
            act0 = [] # feL 2p
            act1 = [77,80,81] # feR 2p
            act2 = [83,84,85,86,87] # feL 3d
            act3 = [] # Sc 3p
            act4 = [94,95,96,97,98] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 4
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 3
        elif 'gs' in model:
            act0 = [] # feL 2p
            act1 = [] # feR 2p
            act2 = [83,84,85,86,87] # feL 3d
            act3 = [88,89,90,91,92,93] # Sc 3p
            act4 = [94,95,96,97,98] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 5
            nb = len(idx) - 4
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 0
        else:
            assert False
    elif '_lunoloc' in model:
        if '22o35e' in model:
            act0 = [11,14,16] # feL 2p
            act1 = [12,13,15] # feR 2p
            act2 = [90,92,93,95,89] # feL 3d
            act3 = [81,82,85,86,79,80] # Sc 3p
            act4 = [98,91,94,96,97] # feR 3d

            sp_sigma_terminal = [75,76,77,78] # feL 3d
            sp_pi_terminal = [83,84,87,88] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 4
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 6
        elif '13o17eL' in model:
            act0 = [11,14,16] # feL 2p
            act2 = [90,92,93,95,89] # feL 3d
            act4 = [98,91,94,96,97] # feR 3d

            idx = act0+act2+act4
            na = len(idx) - 4
            nb = len(idx) - 5
            act_2p = act0
            act_3d = act2+act4
            n_pcore = 3
        elif '13o17eR' in model:
            act1 = [12,13,15] # feR 2p
            act2 = [90,92,93,95,89] # feL 3d
            act4 = [98,91,94,96,97] # feR 3d

            idx = act1+act2+act4
            na = len(idx) - 4
            nb = len(idx) - 5
            act_2p = act1
            act_3d = act2+act4
            n_pcore = 3
        else:
            assert False
    elif '_dft' in model:
        if '22o35e' in model:
            act0 = [38,39,42] # feL 2p
            act1 = [37,40,41] # feR 2p
            #act0 = [] # feL 2p
            #act1 = [] # feR 2p
            act2 = [83,84,85,86,87] # feL 3d
            act3 = [88,89,90,91,92,93] # Sc 3p
            act4 = [94,95,96,97,98] # feR 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 4
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 6
    elif '_lunocloc' in model:
        if '22o35e' in model:
            act0 = [11,12,15] # feL 2p
            act1 = [14,16,17] # feR 2p
            act2 = [89,90,91,93,96] # feL 3d
            act3 = [88,87,86,85,84,83] # Sc 3p
            act4 = [92,94,95,97,98] # feR 3d

            sp_sigma_terminal = [79,80,81,82] # feL 3d
            idx = act0+act1+act2+act3+act4
            na = len(idx) - 4
            nb = len(idx) - 5
            act_2p = act0+act1
            act_3d = act2+act4
            n_pcore = 6
        elif '13o17eL' in model:
            act0 = [11,12,15] # feL 2p
            act2 = [89,90,91,93,96] # feL 3d
            act4 = [92,94,95,97,98] # feR 3d

            idx = act0+act2+act4
            na = len(idx) - 4
            nb = len(idx) - 5
            act_2p = act0
            act_3d = act2+act4
            n_pcore = 3
        elif '13o17eR' in model:
            act1 = [14,16,17] # feR 2p
            act2 = [89,90,91,93,96] # feL 3d
            act4 = [92,94,95,97,98] # feR 3d

            idx = act1+act2+act4
            na = len(idx) - 4
            nb = len(idx) - 5
            act_2p = act1
            act_3d = act2+act4
            n_pcore = 3
        else:
            assert False
    else:
        assert False
else:
    assert False


c_list = list(set(list(range(1,98))) - set(act_list))
clmo = lmo[:, numpy.array(c_list) - 1].copy() 
almo = lmo[:, numpy.array(act_list) - 1].copy() 
vlmo = lmo[:, 98:].copy() 

from pyscf import lo
clmo = lo.PM(mol, clmo).kernel()
almo = lo.PM(mol, almo).kernel()

lcmo = numpy.hstack((clmo, almo, vlmo)).copy()

# Lowdin Population Analysis and Dump lunoloc
from pyxray.utils.addons import lowdinPop
lowdinPop(mol, lcmo)

f = h5py.File('%s.h5' % model, 'w')
g = f.create_group('luno')
g.create_dataset('mo_coeff', data=lcmo)
f.close()

with open('lunocloc.molden','w') as thefile:
    molden.header(mol, thefile)
    molden.orbital_coeff(mol, thefile, lcmo)

