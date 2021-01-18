lake = []

with open("lake.dat", "r") as file:
    for line in file:
        lake.append(list(line.rstrip()))

for l in lake:
    print(l)
