# EntropyBoundConnectionLearning
This tests an original Idea using spiking nerual networks and reinforcment learning


```
numInputs = <hyper_param>
numOutputs = <hyper_param>
startNodes = <hyper_param>
maxLinks = <hyper_param>

functionList = <hyper_param>
threshholdMin = <hyper_param>
threshholdMax = <hyper_param>
leakRateMax = <hyper_param>
leakRateMin = <hyper_param>
weightMax = <hyper_param>
weightMin = <hyper_param>

Network = {}
for i in numInputs+numOutputs+startNodes:
  Network.nodes[i].currentActivation = 0
  Network.nodes[i].activationFuction = RandomSelection(functionList)
  Network.nodes[i].threshold = RandomSelection(threshholdMin, threshholdMax)
  Network.nodes[i].leakRate = RandomSelection(leakRateMin, leakRateMax)
  Network.nodes[i].isActivated = False

  if i < numInputs:
    Network.nodes[i].input = True
  else if i < numOutputs:
    Network.nodes[i].output = True

  for each i,j in Network.nodes:
    if i < numInputs:
      continue
    else if j < numOutputs:
      continue
    Network.links[i][j].probability = 0.5
    Network.links[i][j].weight = RandomSelection(weightMin, weightMax)

while Network.activeLinks.count < maxLinks:
    Network.activeLinks.add(RandomSelectionByEntropy(Network.links))

for link in Network.links:
  if link in Network.activeLinks:
    Network.aliveLinks.add(link)
  else:
    Network.deadLinks.add(link)

while True:
  

```
