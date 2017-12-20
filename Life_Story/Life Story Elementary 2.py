def cellular_automaton(origin, rule, generation):
    codex = {0: '___', 1: '__X', 2: '_X_', 3: '_XX', 4: 'X__', 5: 'X_X', 6: 'XX_', 7: 'XXX'}
    evolution = {}
    for n in range (0, 8):
        if rule[n] == 0:
            evolution[codex[n]] = '_'
        else:
            evolution[codex[n]] = 'X'

    for g in range (0, generation):
        vitality = origin.count('X', 0, len(origin))
        if g < 10:
            print('Gen 0'+str(g)+': '+origin+' '+str(vitality))
        else:
            print('Gen ' + str(g) + ': ' + origin+' '+str(vitality))
        evolved = ''
        for n in range (0, len(origin)):
            cell = origin[n - 1] + origin[n] + origin[(n + 1) % len(origin)]
            evolved += evolution[cell]
        origin = evolved
        g += 1

origin = input(
    'Please input origin ecosystem:\n'
    'e.g. ___________________________________________________________________________________________________X___________________________________________________________________________________________________\n')
while True:
    print ('Origin:   '+origin)
    destiny = str(input(
        'Please choose the destiny:\n'
        'e.g. 00000001\n'
        'Codex:  1: [ ___ ] 2: [ __X ] 3: [ _X_ ] 4: [ _XX ] 5: [ X__ ] 6: [ X_X ] 7: [ XX_ ] 8: [ XXX ]\n'
    ))
    rule = []
    for pattern in destiny:
        rule.append(int(pattern))
    cellular_automaton(origin, rule, 100)
    print ('\nNext Destiny\n')
