"""
Build the two reciprocal ring x loop chimeras of microcin J25 and klebsidin by
grafting one peptide's loop onto the other's ring, using PyMOL CA-only pair_fit.

The upper plug is treated as part of the loop:
  microcin J25   ring 1-8   loop 9-19 (YFVGIGTPISF)   lower plug 20   tail 21
  klebsidin      ring 1-8   loop 9-17 (FFNPNGVMH)     lower plug 18   tail 19

  chimera1  klebsidin ring   + microcin J25 loop  -> GSDGPIIEYFVGIGTPISFYG (21 res)
  chimera2  microcin J25 ring + klebsidin loop    -> GGAGHVPEFFNPNGVMHYG   (19 res)

Run:  pymol -cq graft_chimera.py -- chimera1
      pymol -cq graft_chimera.py -- chimera2
"""
import sys
from pymol import cmd

# User parameters
pwd = ''

MCC  = f'{pwd}microcinJ25_wt_peptide.pdb'
KLEB = f'{pwd}klebsidin_wt_peptide.pdb'

BUILDS = {
    # scaffold donor is superposed onto the loop donor's frame over ring + lower plug/tail
    'chimera1': dict(
        ring_obj='kleb', ring_sel='resi 1-8',
        loop_obj='mcc',  loop_sel='resi 9-19',
        plug_obj='kleb', plug_sel='resi 18-19', plug_shift=+2,   # 18-19 -> 20-21
        mobile='mcc  and name CA and resi 1-8+20-21',
        target='kleb and name CA and resi 1-8+18-19',
        out='MccJ25loop_klebsidin_chimera_heavy.pdb',
    ),
    'chimera2': dict(
        ring_obj='mcc',  ring_sel='resi 1-8',
        loop_obj='kleb', loop_sel='resi 9-17',
        plug_obj='mcc',  plug_sel='resi 20-21', plug_shift=-2,   # 20-21 -> 18-19
        mobile='kleb and name CA and resi 1-8+18-19',
        target='mcc  and name CA and resi 1-8+20-21',
        out='klebsidinloop_mccj25_chimera_heavy.pdb',
    ),
}

which = sys.argv[1] if len(sys.argv) > 1 else 'chimera1'
b = BUILDS[which]

cmd.load(MCC,  'mcc')
cmd.load(KLEB, 'kleb')

rms = cmd.pair_fit(b['mobile'], b['target'])
print('scaffold CA pair_fit RMS = %.3f A' % rms)

cmd.create('p_ring', f"{b['ring_obj']} and {b['ring_sel']}")
cmd.create('p_loop', f"{b['loop_obj']} and {b['loop_sel']}")
cmd.create('p_plug', f"{b['plug_obj']} and {b['plug_sel']}")
cmd.alter('p_plug', f"resi=str(int(resi)+({b['plug_shift']}))")
cmd.sort()

cmd.create('chim', 'p_ring p_loop p_plug')
cmd.alter('chim', "chain='A'")
cmd.alter('chim', "segi=''")
cmd.remove('chim and hydro')
cmd.sort()
cmd.set('pdb_use_ter_records', 0)
cmd.set('retain_order', 0)
cmd.save(b['out'], 'chim')

seq = cmd.get_fastastr('chim').splitlines()
print(which, ''.join(s for s in seq if not s.startswith('>')))
print('wrote', b['out'])
