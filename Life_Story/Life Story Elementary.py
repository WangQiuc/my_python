def cellular_automaton(origin, rule, generation):
    codex = {1:'___', 2:'__X', 4:'_X_', 8:'_XX', 16:'X__', 32:'X_X', 64:'XX_', 128:'XXX'}
    evolution = {}
    for n in range (7, -1, -1):
        if rule // (2**n) == 1:
            rule = rule % (2**n)
            evolution[codex[2**n]] = 'X'
        else:
            evolution[codex[2**n]] = '_'

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

# origin = input('Please input origin cell: ')
while True:
    print ('Codex:  1:[ ___ ] 2:[ __X ] 4:[ _X_ ] 8:[ _XX ] 16:[ X__ ] 32:[ X_X ] 64:[ XX_ ] 128:[ XXX ]')
    origin = '___________________________________________________________________________________________________X___________________________________________________________________________________________________'
    print ('Origin:   '+origin)
    rule = int(input('Please choose the destiny (0-255):\n'))
    cellular_automaton(origin, rule, 100)
    print ('\nNext Destiny\n')