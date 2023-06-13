import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import click
import numpy as np
from six.moves import cPickle

from hominid_pipeline import hominid, utils, gia_utils
import tensorflow as tf


def gia_experiments(model, x_test, y_test, gia_path):

    alphabet = 'ACGT'
    gia = gia_utils.GlobalImportance(model, alphabet)
    gia.set_null_model('random', base_sequence=x_test, num_sample=2000, base_scores=y_test[:,0])
    gia.filter_null(low=10, high=90, num_sample=1000)

    # position = 125
    # class_index = 0
    #
    # print('ap1')
    # ap1 = 'NNTCACGCANN'
    # ap1_flank_scores, ap1_flanks = gia.optimal_flanks(ap1, position, class_index)
    # ap1 = ap1_flanks[0]
    #
    # print('dre')
    # dre = 'NNTATCGATATANN'
    # dre_flank_scores, dre_flanks = gia.optimal_flanks(dre, position, class_index)
    # dre = dre_flanks[0]
    #
    # print('dref')
    # dref = 'NNCGATGGNN'
    # dref_flank_scores, dref_flanks = gia.optimal_flanks(dref, position, class_index)
    # dref = dref_flanks[0]
    #
    # print('gata')
    # gata = 'NNGATAANN'
    # gata_flank_scores, gata_flanks = gia.optimal_flanks(gata, position, class_index)
    # gata = gata_flanks[0]
    #
    # print('gata_rc')
    # gata_rc = 'NNTTATCNN'
    # gata_rc_flank_scores, gata_rc_flanks = gia.optimal_flanks(gata_rc, position, class_index)
    # gata_rc = gata_rc_flanks[0]
    #
    # print('ohler5')
    # ohler5 = 'NNCAGCTGNN'
    # ohler5_flank_scores, ohler5_flanks = gia.optimal_flanks(ohler5, position, class_index)
    # ohler5 = ohler5_flanks[0]
    #
    # print('ohler7')
    # ohler7 = 'NNCATCGCTGNN'
    # ohler7_flank_scores, ohler7_flanks = gia.optimal_flanks(ohler7, position, class_index)
    # ohler7 = ohler7_flanks[0]
    #
    # print('mitf')
    # mitf = 'NNTCACGTGANN'
    # mitf_flank_scores, mitf_flanks = gia.optimal_flanks(mitf, position, class_index)
    # mitf = mitf_flanks[0]
    #
    # print('ohler1')
    # ohler1 = 'NCAGTGTGACCGN'
    # ohler1_flank_scores, ohler1_flanks = gia.optimal_flanks(ohler1, position, class_index)
    # ohler1 = ohler1_flanks[0]
    #
    # print('ohler1_rc')
    # ohler1_rc = 'NCGGTCACACTGN'
    # ohler1_rc_flank_scores, ohler1_rc_flanks = gia.optimal_flanks(ohler1_rc, position, class_index)
    # ohler1_rc = ohler1_rc_flanks[0]
    #
    #
    # flank_scores0 = {
    #     'ap1': ap1_flank_scores,
    #     'dre': dre_flank_scores,
    #     'dref': dref_flank_scores,
    #     'gata': gata_flank_scores,
    #     'gata_rc': gata_rc_flank_scores,
    #     'ohler5': ohler5_flank_scores,
    #     'ohler7': ohler7_flank_scores,
    #     'mitf': mitf_flank_scores,
    #     'ohler1': ohler1_flank_scores,
    #     'ohler1_rc': ohler1_rc_flank_scores,
    # }
    #
    # flanks0 = {
    #     'ap1': ap1_flanks,
    #     'dre': dre_flanks,
    #     'dref': dref_flanks,
    #     'gata': gata_flanks,
    #     'gata_rc': gata_rc_flanks,
    #     'ohler5': ohler5_flanks,
    #     'ohler7': ohler7_flanks,
    #     'mitf': mitf_flanks,
    #     'ohler1': ohler1_flanks,
    #     'ohler1_rc': ohler1_rc_flanks,
    # }
    #
    # motifs0 = {
    #     'ap1': ap1,
    #     'dre': dre,
    #     'dref': dref,
    #     'gata': gata,
    #     'gata_rc': gata_rc,
    #     'ohler5': ohler5,
    #     'ohler7': ohler7,
    #     'mitf': mitf,
    #     'ohler1': ohler1,
    #     'ohler1_rc': ohler1_rc,
    #     'ohler6': 'AAAATACCAAA',
    #     'ohler6_rc': 'TTTGGTATTTTT',
    #     'stat': 'TTCCCGGAA',
    #     'ts01': 'AAATTTAAAA',
    #     'cebp': 'TGGTGAAAT',
    #     'at': 'AATAAA',
    #     'ga': 'GAGAGAGAGAGA',
    #     'ct': 'CTCTCTCTCTCT',
    #     'myb': 'CGCG',
    #     'context': 'GGGCT',
    # }
    #
    #
    # position = 125
    # class_index = 1
    #
    # print('ap1')
    # ap1 = 'NNTCACGCANN'
    # ap1_flank_scores, ap1_flanks = gia.optimal_flanks(ap1, position, class_index)
    # ap1 = ap1_flanks[0]
    #
    # print('dre')
    # dre = 'NNTATCGATATANN'
    # dre_flank_scores, dre_flanks = gia.optimal_flanks(dre, position, class_index)
    # dre = dre_flanks[0]
    #
    # print('dref')
    # dref = 'NNCGATGGNN'
    # dref_flank_scores, dref_flanks = gia.optimal_flanks(dref, position, class_index)
    # dref = dref_flanks[0]
    #
    # print('gata')
    # gata = 'NNGATAANN'
    # gata_flank_scores, gata_flanks = gia.optimal_flanks(gata, position, class_index)
    # gata = gata_flanks[0]
    #
    # print('gata_rc')
    # gata_rc = 'NNTTATCNN'
    # gata_rc_flank_scores, gata_rc_flanks = gia.optimal_flanks(gata_rc, position, class_index)
    # gata_rc = gata_rc_flanks[0]
    #
    # print('ohler5')
    # ohler5 = 'NNCAGCTGNN'
    # ohler5_flank_scores, ohler5_flanks = gia.optimal_flanks(ohler5, position, class_index)
    # ohler5 = ohler5_flanks[0]
    #
    # print('ohler7')
    # ohler7 = 'NNCATCGCTGNN'
    # ohler7_flank_scores, ohler7_flanks = gia.optimal_flanks(ohler7, position, class_index)
    # ohler7 = ohler7_flanks[0]
    #
    # print('mitf')
    # mitf = 'NNTCACGTGANN'
    # mitf_flank_scores, mitf_flanks = gia.optimal_flanks(mitf, position, class_index)
    # mitf = mitf_flanks[0]
    #
    # print('ohler1')
    # ohler1 = 'NCAGTGTGACCGN'
    # ohler1_flank_scores, ohler1_flanks = gia.optimal_flanks(ohler1, position, class_index)
    # ohler1 = ohler1_flanks[0]
    #
    # print('ohler1_rc')
    # ohler1_rc = 'NCGGTCACACTGN'
    # ohler1_rc_flank_scores, ohler1_rc_flanks = gia.optimal_flanks(ohler1_rc, position, class_index)
    # ohler1_rc = ohler1_rc_flanks[0]
    #
    #
    # flank_scores1 = {
    #     'ap1': ap1_flank_scores,
    #     'dre': dre_flank_scores,
    #     'dref': dref_flank_scores,
    #     'gata': gata_flank_scores,
    #     'gata_rc': gata_rc_flank_scores,
    #     'ohler5': ohler5_flank_scores,
    #     'ohler7': ohler7_flank_scores,
    #     'mitf': mitf_flank_scores,
    #     'ohler1': ohler1_flank_scores,
    #     'ohler1_rc': ohler1_rc_flank_scores,
    # }
    #
    # flanks1 = {
    #     'ap1': ap1_flanks,
    #     'dre': dre_flanks,
    #     'dref': dref_flanks,
    #     'gata': gata_flanks,
    #     'gata_rc': gata_rc_flanks,
    #     'ohler5': ohler5_flanks,
    #     'ohler7': ohler7_flanks,
    #     'mitf': mitf_flanks,
    #     'ohler1': ohler1_flanks,
    #     'ohler1_rc': ohler1_rc_flanks,
    # }
    #
    # motifs1 = {
    #     'ap1': ap1,
    #     'dre': dre,
    #     'dref': dref,
    #     'gata': gata,
    #     'gata_rc': gata_rc,
    #     'ohler5': ohler5,
    #     'ohler7': ohler7,
    #     'mitf': mitf,
    #     'ohler1': ohler1,
    #     'ohler1_rc': ohler1_rc,
    #     'ohler6': 'AAAATACCAAA',
    #     'ohler6_rc': 'TTTGGTATTTTT',
    #     'stat': 'TTCCCGGAA',
    #     'ts01': 'AAATTTAAAA',
    #     'cebp': 'TGGTGAAAT',
    #     'at': 'AATAAA',
    #     'ga': 'GAGAGAGAGAGA',
    #     'ct': 'CTCTCTCTCTCT',
    #     'myb': 'CGCG',
    #     'context': 'GGGCT',
    # }
    #
    # start = 50
    # end = 200
    # class_index = 0
    #
    # print('ap1')
    # ap1_pos_scores, ap1_pos = gia.optimal_position(motifs0['ap1'], start, end, class_index)
    # ap1_index = ap1_pos[np.argmax(np.mean(ap1_pos_scores, axis=1))]
    #
    # print('dre')
    # dre_pos_scores, dre_pos = gia.optimal_position(motifs0['dre'], start, end, class_index)
    # dre_index = dre_pos[np.argmax(np.mean(dre_pos_scores, axis=1))]
    #
    # print('dref')
    # dref_pos_scores, dref_pos = gia.optimal_position(motifs0['dref'], start, end, class_index)
    # dref_index = dref_pos[np.argmax(np.mean(dref_pos_scores, axis=1))]
    #
    # print('gata')
    # gata_pos_scores, gata_pos = gia.optimal_position(motifs0['gata'], start, end, class_index)
    # gata_index = gata_pos[np.argmax(np.mean(gata_pos_scores, axis=1))]
    #
    # print('gata_rc')
    # gata_rc_pos_scores, gata_rc_pos = gia.optimal_position(motifs0['gata_rc'], start, end, class_index)
    # gata_rc_index = gata_rc_pos[np.argmax(np.mean(gata_rc_pos_scores, axis=1))]
    #
    # print('ohler5')
    # ohler5_pos_scores, ohler5_pos = gia.optimal_position(motifs0['ohler5'], start, end, class_index)
    # ohler5_index = ohler5_pos[np.argmax(np.mean(ohler5_pos_scores, axis=1))]
    #
    # print('ohler7')
    # ohler7_pos_scores, ohler7_pos = gia.optimal_position(motifs0['ohler7'], start, end, class_index)
    # ohler7_index = ohler7_pos[np.argmax(np.mean(ohler7_pos_scores, axis=1))]
    #
    # print('mitf')
    # mitf_pos_scores, mitf_pos = gia.optimal_position(motifs0['mitf'], start, end, class_index)
    # mitf_index = mitf_pos[np.argmax(np.mean(mitf_pos_scores, axis=1))]
    #
    # print('ohler1')
    # ohler1_pos_scores, ohler1_pos = gia.optimal_position(motifs0['ohler1'], start, end, class_index)
    # ohler1_index = ohler1_pos[np.argmax(np.mean(ohler1_pos_scores, axis=1))]
    #
    # print('ohler1_rc')
    # ohler1_rc_pos_scores, ohler1_rc_pos = gia.optimal_position(motifs0['ohler1_rc'], start, end, class_index)
    # ohler1_rc_index = ohler1_rc_pos[np.argmax(np.mean(ohler1_rc_pos_scores, axis=1))]
    #
    # print('ohler6')
    # ohler6_pos_scores, ohler6_pos = gia.optimal_position(motifs0['ohler6'], start, end, class_index)
    # ohler6_index = ohler6_pos[np.argmax(np.mean(ohler6_pos_scores, axis=1))]
    #
    # print('ohler6_rc')
    # ohler6_rc_pos_scores, ohler6_rc_pos = gia.optimal_position(motifs0['ohler6_rc'], start, end, class_index)
    # ohler6_rc_index = ohler6_rc_pos[np.argmax(np.mean(ohler6_rc_pos_scores, axis=1))]
    #
    # print('stat')
    # stat_pos_scores, stat_pos = gia.optimal_position(motifs0['stat'], start, end, class_index)
    # stat_index = stat_pos[np.argmax(np.mean(stat_pos_scores, axis=1))]
    #
    # print('ts01')
    # ts01_pos_scores, ts01_pos = gia.optimal_position(motifs0['ts01'], start, end, class_index)
    # ts01_index = ts01_pos[np.argmax(np.mean(ts01_pos_scores, axis=1))]
    #
    # print('cebp')
    # cebp_pos_scores, cebp_pos = gia.optimal_position(motifs0['cebp'], start, end, class_index)
    # cebp_index = cebp_pos[np.argmax(np.mean(cebp_pos_scores, axis=1))]
    #
    # pos_scores0 = {
    #     'ap1': ap1_pos_scores,
    #     'dre': dre_pos_scores,
    #     'dref': dref_pos_scores,
    #     'gata': gata_pos_scores,
    #     'gata_rc': gata_rc_pos_scores,
    #     'ohler5': ohler5_pos_scores,
    #     'ohler7': ohler7_pos_scores,
    #     'mitf': mitf_pos_scores,
    #     'ohler1': ohler1_pos_scores,
    #     'ohler1_rc': ohler1_rc_pos_scores,
    #     'ohler6': ohler6_pos_scores,
    #     'ohler6_rc': ohler6_rc_pos_scores,
    #     'stat': stat_pos_scores,
    #     'ts01': ts01_pos_scores,
    #     'cebp': cebp_pos_scores,
    # }
    #
    # pos_index0 = {
    #     'ap1': ap1_index,
    #     'dre': dre_index,
    #     'dref': dref_index,
    #     'gata': gata_index,
    #     'gata_rc': gata_rc_index,
    #     'ohler5': ohler5_index,
    #     'ohler7': ohler7_index,
    #     'mitf': mitf_index,
    #     'ohler1': ohler1_index,
    #     'ohler1_rc': ohler1_rc_index,
    #     'ohler6': ohler6_index,
    #     'ohler6_rc': ohler6_rc_index,
    #     'stat': stat_index,
    #     'ts01': ts01_index,
    #     'cebp': cebp_index,
    # }
    #
    # start = 50
    # end = 200
    # class_index = 1
    #
    # print('ap1')
    # ap1_pos_scores, ap1_pos = gia.optimal_position(motifs1['ap1'], start, end, class_index)
    # ap1_index = ap1_pos[np.argmax(np.mean(ap1_pos_scores, axis=1))]
    #
    # print('dre')
    # dre_pos_scores, dre_pos = gia.optimal_position(motifs1['dre'], start, end, class_index)
    # dre_index = dre_pos[np.argmax(np.mean(dre_pos_scores, axis=1))]
    #
    # print('dref')
    # dref_pos_scores, dref_pos = gia.optimal_position(motifs1['dref'], start, end, class_index)
    # dref_index = dref_pos[np.argmax(np.mean(dref_pos_scores, axis=1))]
    #
    # print('gata')
    # gata_pos_scores, gata_pos = gia.optimal_position(motifs1['gata'], start, end, class_index)
    # gata_index = gata_pos[np.argmax(np.mean(gata_pos_scores, axis=1))]
    #
    # print('gata_rc')
    # gata_rc_pos_scores, gata_rc_pos = gia.optimal_position(motifs1['gata_rc'], start, end, class_index)
    # gata_rc_index = gata_rc_pos[np.argmax(np.mean(gata_rc_pos_scores, axis=1))]
    #
    # print('ohler5')
    # ohler5_pos_scores, ohler5_pos = gia.optimal_position(motifs1['ohler5'], start, end, class_index)
    # ohler5_index = ohler5_pos[np.argmax(np.mean(ohler5_pos_scores, axis=1))]
    #
    # print('ohler7')
    # ohler7_pos_scores, ohler7_pos = gia.optimal_position(motifs1['ohler7'], start, end, class_index)
    # ohler7_index = ohler7_pos[np.argmax(np.mean(ohler7_pos_scores, axis=1))]
    #
    # print('mitf')
    # mitf_pos_scores, mitf_pos = gia.optimal_position(motifs1['mitf'], start, end, class_index)
    # mitf_index = mitf_pos[np.argmax(np.mean(mitf_pos_scores, axis=1))]
    #
    # print('ohler1')
    # ohler1_pos_scores, ohler1_pos = gia.optimal_position(motifs1['ohler1'], start, end, class_index)
    # ohler1_index = ohler1_pos[np.argmax(np.mean(ohler1_pos_scores, axis=1))]
    #
    # print('ohler1_rc')
    # ohler1_rc_pos_scores, ohler1_rc_pos = gia.optimal_position(motifs1['ohler1_rc'], start, end, class_index)
    # ohler1_rc_index = ohler1_rc_pos[np.argmax(np.mean(ohler1_rc_pos_scores, axis=1))]
    #
    # print('ohler6')
    # ohler6_pos_scores, ohler6_pos = gia.optimal_position(motifs1['ohler6'], start, end, class_index)
    # ohler6_index = ohler6_pos[np.argmax(np.mean(ohler6_pos_scores, axis=1))]
    #
    # print('ohler6_rc')
    # ohler6_rc_pos_scores, ohler6_rc_pos = gia.optimal_position(motifs1['ohler6_rc'], start, end, class_index)
    # ohler6_rc_index = ohler6_rc_pos[np.argmax(np.mean(ohler6_rc_pos_scores, axis=1))]
    #
    # print('stat')
    # stat_pos_scores, stat_pos = gia.optimal_position(motifs1['stat'], start, end, class_index)
    # stat_index = stat_pos[np.argmax(np.mean(stat_pos_scores, axis=1))]
    #
    # print('ts01')
    # ts01_pos_scores, ts01_pos = gia.optimal_position(motifs1['ts01'], start, end, class_index)
    # ts01_index = ts01_pos[np.argmax(np.mean(ts01_pos_scores, axis=1))]
    #
    # print('cebp')
    # cebp_pos_scores, cebp_pos = gia.optimal_position(motifs1['cebp'], start, end, class_index)
    # cebp_index = cebp_pos[np.argmax(np.mean(cebp_pos_scores, axis=1))]
    #
    # pos_scores1 = {
    #     'ap1': ap1_pos_scores,
    #     'dre': dre_pos_scores,
    #     'dref': dref_pos_scores,
    #     'gata': gata_pos_scores,
    #     'gata_rc': gata_rc_pos_scores,
    #     'ohler5': ohler5_pos_scores,
    #     'ohler7': ohler7_pos_scores,
    #     'mitf': mitf_pos_scores,
    #     'ohler1': ohler1_pos_scores,
    #     'ohler1_rc': ohler1_rc_pos_scores,
    #     'ohler6': ohler6_pos_scores,
    #     'ohler6_rc': ohler6_rc_pos_scores,
    #     'stat': stat_pos_scores,
    #     'ts01': ts01_pos_scores,
    #     'cebp': cebp_pos_scores,
    # }
    #
    # pos_index1 = {
    #     'ap1': ap1_index,
    #     'dre': dre_index,
    #     'dref': dref_index,
    #     'gata': gata_index,
    #     'gata_rc': gata_rc_index,
    #     'ohler5': ohler5_index,
    #     'ohler7': ohler7_index,
    #     'mitf': mitf_index,
    #     'ohler1': ohler1_index,
    #     'ohler1_rc': ohler1_rc_index,
    #     'ohler6': ohler6_index,
    #     'ohler6_rc': ohler6_rc_index,
    #     'stat': stat_index,
    #     'ts01': ts01_index,
    #     'cebp': cebp_index,
    # }
    #
    # pairs = [
    #     #'ohler1_rc-dre',
    #     #'dre-ohler1_rc',
    #     'dre-ohler1',
    #     'ohler1_rc-ohler6_rc',
    #     'ohler7-dre',
    #     'dre-ohler7',
    #     'ohler7-dref',
    #     'dref-ohler7',
    #     'gata-ap1',
    #     'gata_rc-ap1',
    #     'ap1-gata',
    #     'ap1-gata_rc',
    #     #'gata-ct',
    #     #'gata_rc-ct',
    #     #'gata-ga',
    #     #'gata_rc-ga',
    #     'ohler1-dre',
    #
    #     'ohler1-ohler6',
    #     'ohler1-ohler6_rc',
    #     'ohler1_rc-ohler6',
    #     'ohler6-ohler1',
    #     'ohler6-ohler1_rc',
    #     'ohler6_rc-ohler1',
    #     'ohler6_rc-ohler1_rc',
    #     'ohler6-dre',
    #     'ohler6_rc-dre',
    #     'stat-ts01',
    #     'ts01-stat',
    #     'cebp-ohler1',
    #     'ohler1-cebp',
    #     'ohler5-mitf',
    #     'mitf-ohler5',
    #     'mitf-ohler1',
    #     'mitf-ohler6',
    #     'mitf-ohler7',
    #     'mitf-gata',
    #     'mitf-stat',
    #     'mitf-cebp',
    #     'gata-context',
    #     'ohler1-context',
    #     'ohler6-context',
    #     'ohler7-context',
    #     'dre-context',
    #     'ts01-myb',
    #     'mitf-myb',
    #     'ohler7-myb',
    #     'dre-ap1',
    #     'ap1-dre',
    #     'ohler1-ap1',
    #     'ohler1_rc-ap1',
    #     'ap1-ohler1',
    #     'ap1-ohler1_rc',
    #     'dre-gata',
    #     'dre-gata_rc',
    #     'gata-dre',
    #     'gata_rc-dre',
    #     'ohler6-at',
    #     'ohler6_rc-at',
    # ]
    #
    #
    # interact_scores0 = {}
    # interact_pos_scores0 = {}
    # for pair in pairs:
    #     print(pair)
    #     motif1, motif2 = pair.split('-')
    #     scores, pos_scores = gia_utils.optimal_interactions(gia, motifs0[motif1], pos_index0[motif1], motifs0[motif2], window_scan=50, class_index=0)
    #     interact_scores0[pair] = scores
    #     interact_pos_scores0[pair] = pos_scores
    #
    #
    # interact_scores1 = {}
    # interact_pos_scores1 = {}
    # for pair in pairs:
    #     print(pair)
    #     motif1, motif2 = pair.split('-')
    #     scores, pos_scores = gia_utils.optimal_interactions(gia, motifs1[motif1], pos_index1[motif1], motifs0[motif2], window_scan=50, class_index=1)
    #     interact_scores1[pair] = scores
    #     interact_pos_scores1[pair] = pos_scores
    #
    #
    # with open(f'{gia_path}/gia_results.pickle', 'wb') as fout:
    #     cPickle.dump(motifs0, fout)
    #     cPickle.dump(flanks0, fout)
    #     cPickle.dump(flank_scores0, fout)
    #     cPickle.dump(pos_scores0, fout)
    #     cPickle.dump(pos_index0, fout)
    #     cPickle.dump(interact_scores0, fout)
    #     cPickle.dump(interact_pos_scores0, fout)
    #     cPickle.dump(motifs1, fout)
    #     cPickle.dump(flanks1, fout)
    #     cPickle.dump(flank_scores1, fout)
    #     cPickle.dump(pos_scores1, fout)
    #     cPickle.dump(pos_index1, fout)
    #     cPickle.dump(interact_scores1, fout)
    #     cPickle.dump(interact_pos_scores1, fout)


    pos_scores, pos = gia.optimal_position('GATAA', start=50, end=200, class_index=0)
    gata_index0 = pos[np.argmax(np.mean(pos_scores, axis=1))]

    pos_scores, pos = gia.optimal_position('TTATC', start=50, end=200, class_index=0)
    gata_rc_index0 = pos[np.argmax(np.mean(pos_scores, axis=1))]


    motif1 = 'TTATC'
    motif2 = 'AGAGAGAG'
    scores, pos_scores = gia_utils.optimal_interactions(gia, motif1, gata_rc_index0, motif2, window_scan=50, class_index=0)

    motif1 = 'TTATC'
    motif2 = 'CTCTCTCT'
    scores2, pos_scores = gia_utils.optimal_interactions(gia, motif1, gata_rc_index0, motif2, window_scan=50, class_index=0)

    motif1 = 'GATAA'
    motif2 = 'AGAGAGAG'
    scores3, pos_scores = gia_utils.optimal_interactions(gia, motif1, gata_index0, motif2, window_scan=50, class_index=0)

    motif1 = 'GATAA'
    motif2 = 'CTCTCTCT'
    scores4, pos_scores = gia_utils.optimal_interactions(gia, motif1, gata_index0, motif2, window_scan=50, class_index=0)


    with open(f'{gia_path}/gata_results.pickle', 'wb') as fout:
        cPickle.dump(scores, fout)
        cPickle.dump(scores2, fout)
        cPickle.dump(scores3, fout)
        cPickle.dump(scores4, fout)

    return




@click.command()
@click.option("--config_file", type=str)
@click.option("--smoke_test", type=bool, default=False)
def main(config_file: str, smoke_test: bool):

    save_path = config_file.split("config.yaml")[0]
    config = hominid.load_config(config_file)

    tuner = hominid.HominidTuner(
                        config,
                        epochs=1,
                        tuning_mode=False,
                        save_path=save_path,
                        subsample=smoke_test
                        )

    print(f"Loading model and dataset!")

    x_test, y_test = tuner.data_processor.load_data("test")
    print(x_test.shape)

    # Build the model
    model = tuner.model_builder.build_model()

    model.compile(
        tf.keras.optimizers.Adam(lr=0.001),
        loss='mse',
        metrics=[utils.Spearman, utils.pearson_r]
        )
    print(model.summary())
    model.load_weights(f'{tuner.save_path}/weights')

    print(f"Performing global importance analysis!")
    gia_path = f'{tuner.save_path}/gia'
    utils.make_directory(gia_path)

    gia_experiments(model, x_test, y_test, gia_path)

    print(f"Finished with GIA experiments!")


if __name__ == "__main__":
    main()
