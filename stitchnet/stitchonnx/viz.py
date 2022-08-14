import seaborn

from graphviz import Digraph

def rgb_to_hex(r, g, b):
    return ('#{:X}{:X}{:X}').format(r, g, b)

seaborn.set_theme()
colors = seaborn.color_palette()
# print(colors)
colors = [rgb_to_hex(int(r*255),int(g*255),int(b*255)) for r,g,b in colors]
# print(colors)
# colors = ["#4C72B0",
# "#DD8452",
# "#55A868",
# "#C44E52",
# "#8172B3",
# "#937860",
# "#DA8BC3",
# "#8C8C8C",
# "#CCB974",
# "#64B5CD"]

def draw_stitchNet(nets, stitchNet, name="_results/stitchnet/net"):
    hexColor = {}
    for i,net in enumerate(nets):
        hexColor[net.get_id()] = colors[i%len(colors)] #rgb_to_hex(*[int(c*255) for c in colors[i%len(colors)]])
        
    dot = Digraph(comment='StitchNet')
    dot.graph_attr['rankdir'] = 'BT'  
    # dot.graph_attr['minlen'] = '0.1'  
    rep = stitchNet.get_id()
    for r in rep:
        dot.node(str(r), f'F{r[1]} of N{r[0]}', fontcolor="white", color=hexColor[r[0]], style="filled", shape="rectangle")
    
    for i,f1 in enumerate(rep[:-1]):
        f2 = rep[i+1]
        dot.edge(str(f1), str(f2), label=str(i))
        # dot.edge(str(f1), str(f2), minlen="1")
            
    dot.format = 'svg'
    dot.render(f'{name}', view=False)
    return dot, f'{name}.{dot.format}'

def draw_stitchNet_fromTuples(fragmentTuples, numNets=10, name="_results/stitchnet/net"):
    hexColor = {}
    for i in range(numNets):
        hexColor[i] = colors[i%len(colors)] #rgb_to_hex(*[int(c*255) for c in colors[i%len(colors)]])
        
    dot = Digraph(comment='StitchNet')
    dot.graph_attr['rankdir'] = 'BT'  
    # dot.graph_attr['minlen'] = '0.1'  
    rep = fragmentTuples
    for r in rep:
        dot.node(str(r), f'F{r[1]} of N{r[0]}', fontcolor="white", color=hexColor[r[0]], style="filled", shape="rectangle")
    
    for i,f1 in enumerate(rep[:-1]):
        f2 = rep[i+1]
        dot.edge(str(f1), str(f2), label=str(i))
        # dot.edge(str(f1), str(f2), minlen="1")
            
    dot.format = 'svg'
    dot.render(f'{name}', view=False)
    return dot, f'{name}.{dot.format}'