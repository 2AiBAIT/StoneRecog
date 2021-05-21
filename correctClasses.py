import json

with open("rocks_db/rocks_db.json") as data_file:
    rocks = json.load(data_file)

with open("rocks_db/rocks_db_V0corrected.json") as data_file:
    corrected_rocks = json.load(data_file)

erradas=[]
erradas_classe=[]
erradas_corrected_classe=[]

for pedra in rocks:
    id=pedra["ID"]
    corrected_pedra_certa=0
    for corrected_pedra in corrected_rocks:
        if corrected_pedra["ID"]==pedra["ID"]:
            corrected_pedra_certa=corrected_pedra
            if corrected_pedra["Classe"]!=pedra["Classe"]:
                erradas.append(pedra["ID"])
                erradas_classe.append(pedra["Classe"])
                erradas_corrected_classe.append(corrected_pedra["Classe"])
                pedra["Classe"] = corrected_pedra["Classe"]
            break

print(erradas)

with open('rocks_db/rocks_db_corrected.json', 'w') as fp:
    json.dump(rocks, fp)