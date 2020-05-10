from sklearn import svm
import numpy as np
from decimal import Decimal

clf = svm.SVC(gamma=0.001, C=100.)
unknown = [['négociants'],['armateurs'],['apparition'],['inouïe'],['cétacé'],['ichtyologistes'],['lieues'],['signalèrent'],['hyperboréennes'],['abîmes'],['polémique'],['crédules'],['ripostait'],['contemporains'],['élucubrations'],['nœuds'],['remous'],['encablures'],['succincts'],['transocéaniennes'],['obstinément'],['m’exécutai'],['abîmes'],['reculés'],['fondrière'],['perforation'],['entamer'],['steamer'],['d’ivoire'],['hallebarde'],['j’opinerais'],['licorne'],['éperon'],['frégates'],['cuirassées'],['nains'],['mollusques'],['quadrumanes'],['échantillons'],['incessamment'],['frégate'],['songeais'],['poursuivre'],['inquiétant'],['pénible'],['ramène'],['digne'],['agrément'],['dévoué'],['phlegmatique'],['embranchements'],['cachalot'],['Formaliste'],['agaçant'],['fébrile'],['préparatifs'],['malle'],['ustensiles'],['crochet'],['capricieuses'],['réglai'],['comptoir'],['vomissait'],['matelots'],['saluai'],['munie'],['aménagements'],['nautiques'],['déplaise'],['buccin'],['arrimer'],['convenablement'],['l’appareillage'],['incrédules'],['rallier'],['gémirent'],['l’hélice'],['majestueusement'],['tenders'],['allongée'],['l’âme'],['juré'],['perroquet'],['corvée'],['matelots'],['étrave'],['hisser'],['dépecer'],['scrupuleuse'],['quiconque'],['mousse'],['matelot'],['barbelées'],['espingoles'],['culasse'],['contrariait'],['aperçoive'],['déduites'],['pisciformes'],['élucider'],['ultérieurement'],['disséquer'],['circonstance'],['merveilleuse'],['baleiniers'],['apprîmes'],['prodigieuse'],['l’unanimité'],['détroit'],['resserré'],['îlot'],['solitaire'],['éblouis'],['nyctalopes'],['faculté'],['navire'],['penché'],['bastingages'],['gaillard'],['cotonneux'],['sillage'],['torrent'],['poitrine '],['haletante'],['bonté'],['écarquiller'],['incrédulité'],['parages'],['prodigieuse'],['parié'],['ébats'],['anévrismes'],['d’éréthisme'],['sillonna'],['brusques'],['flots'],['brèche'],['ardents'],['détracteurs'],['soutiers'],['zèle'],['d’obstination'],['fiévreuse'],['sommation'],['panne'],['étincelait'],['ténèbres'],['maîtres'],['matelots'],['mousses'],['erre'],['aperçûmes'],['hanche'],['tribord'],['l’insoutenable'],['agglomération'],['pholades'],['salpes'],['sienne'],['haletants'],['stupéfaction'],['crainte'],['muets'],['tourbillons'],['brusquement'],['précintes'],['l’eût'],['imprudemment'],['gymnote'],['foudroyante'],['gré'],['assourdissant'],['dunette'],['d’avides'],['vigoureux'],['talons'],['disparition'],['embarcation'],['l’abîme'],['vigoureuse'],['Nullement'],['fâcheuse'],['L’imperturbable'],['chape'],['débarrassa'],['gouvernail'],['embarcations'],['Étonnante'],['embarcations'],['compagnons'],['sentant'],['frisson'],['imprégnés'],['cramponner'],['indignation'],['rôtissoire'],['muraille']]
known = [['Bien'],['que'],['j’eusse'],['été'],['surpris'],['par'],['cette'],['chute'],['inattendue,'],['je'],['n’en'],['pas'],['moins'],['une'],['impression'],['très-nette'],['de'],['mes'],['sensations.'],['Je'],['fus'],['d’abord'],['entraîné'],['à'],['une'],['profondeur'],['de'],['vingt'],['pieds'],['environ.'],['Je'],['suis'],['bon'],['nageur,'],['sans'],['égaler'],['Byron'],['et'],['Edgar'],['Poe,'],['qui'],['sont'],['des'],['maîtres,'],['et'],['ce'],['ne'],['me'],['fit'],['point'],['perdre'],['la'],['tête.'],['Deux'],['vigoureux'],['coups'],['de'],['me'],['ramenèrent'],['à'],['la'],['surface'],['de'],['la'],['mer.'],['Mon'],['premier'],['soin'],['de'],['chercher'],['des'],['yeux'],['la'],['frégate.'],['L’équipage'],['s’était-il'],['aperçu'],['de'],['ma'],['disparition'],['L’Abraham-Lincoln'],['avait-il'],['de'],['bord'],['Le'],['commandant'],['Farragut'],['mettait-il'],['une'],['à'],['la'],['mer'],['Devais-je'],['espérer'],['d’être'],['sauvé'],['Les'],['ténèbres'],['étaient'],['profondes.'],['J’entrevis'],['une'],['masse'],['noire'],['qui'],['disparaissait'],['vers'],['l’est,'],['et'],['dont'],['les'],['de'],['position'],['dans'],['l’éloignement.'],['C’était'],['la'],['frégate.'],['Je'],['me']]

allwords = unknown + known
#allwords = [[[ord(c) for c in y] for y in x] for x in allwords]
allints = allwords
allwords = [[[str(ord(c)) for c in y] for y in x] for x in allwords]


for i in range(len(allwords)):
    allints[i][0] = ''
    for asc in allwords[i][0]:
        allints[i][0] = allints[i][0].join(asc)


labels = []

allints = [[float(y) for y in x] for x in allints]

for i in unknown:
    labels.append(1)

for i in known:
    labels.append(0)

clf.fit(allints, labels)