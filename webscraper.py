from bs4 import BeautifulSoup
import imagesize
import requests
import json


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step


'''TXT Variables'''
na = "N/A"
datasetPath = "D:/Pedras/"

'''
VALUES VARIABLES
rockname - Nome da Pedra
rock_class - Class da Pedra
color - Cor da Pedra
place1 - Localização da Pedra
place2 - Localização da Pedra
trade - Nome de comercialização
pq - Qualidade do Polimento
pd - Durabilidade do Polimento
sff - Se serve para chãos
ar - Resistencia ao acido
bulk_value - Densidade aparente
pres_value - Resistencia a pressao
poro_value - Porosidade
fros_value - Resistencia ao frio
insk1 - 
insk2 - 
'''


def get_rock_details():
    global width, height
    for num in my_range(1, 6556, 1):
        print("Pedra:", str(num), "/", str(6556))
        url = requests.get('https://www.naturalstone-online.com/index.php?id=356&user_dnsaeng_pi1[steinID]=%d' % num).text
        soup = BeautifulSoup(url, 'lxml')
        summary = soup.find('div', class_='detailansicht textcolumns')

        if summary.find('img', class_='stein_bild_big'):
            if summary.find('img', class_='stein_bild_big').attrs['src']:
                img_link = summary.find('img', class_='stein_bild_big').attrs['src']

                file_path = (datasetPath + '%d.jpg' % num)
                imageURL='https://www.naturalstone-online.com' + img_link

                img_data = requests.get(imageURL).content
                with open(file_path, 'wb') as handler:
                    handler.write(img_data)
                # Get image dimensions width and height
                width, height = imagesize.get(file_path)
                #print(width, height)

            '''Name of the Rock'''
            if summary.find('div', class_='box_detailheader').h2:
                rock_name = summary.find('div', class_='box_detailheader').h2.text
                #print(rock_name)
            else:
                rock_name = na
                #print(na)

            '''Rock Class Name'''
            if summary.find('div', class_='sub_img').h3:
                rock_class = summary.find('div', class_='sub_img').h3.text
                #print(rock_class)
            else:
                rock_class = na
                #print(na)

            '''Basic Properties'''
            if summary.find('strong', text='Coloring:'):
                color = summary.find('div', class_='detail_info').span.text
                #print(color)
            else:
                color = na
                #print(na)

            #Pais e Regiao da Pedra
            place1 = na
            place2 = na
            location = summary.find('strong', text='Location where found:')
            if location:
                place1 = location.next_sibling.next_sibling

                location_country = location.next_sibling.next_sibling.next_sibling.next_sibling
                if location_country is not None and location_country.name != 'unknown':
                    place2 = location_country

            if summary.find('strong', text='Trading name:'):
                trade = summary.find('strong', text='Trading name:').next_element.next_element.next_element
                #print(trade)
            else:
                trade = na
                #print(na)

            '''Rock Technical Properties'''
            if summary.find('strong', text='Polish quality:'):
                pq = summary.find('strong', text='Polish quality:').next_element.next_element.next_element
                #print(pq)
            else:
                pq = na
                #print(na)

            if summary.find('strong', text='Polish durability:'):
                pd = summary.find('strong', text='Polish durability:').next_element.next_element.next_element
                #print(pd)
            else:
                pd = na
                #print(na)

            if summary.find('strong', text='Suitable for flooring:'):
                sff = summary.find('strong', text='Suitable for flooring:').next_element.next_element.next_element
                #print(sff)
            else:
                sff = na
                #print(na)

            if summary.find('strong', text='Acid resistant:'):
                ar = summary.find('strong', text='Acid resistant:').next_element.next_element.next_element
                #print(ar)
            else:
                ar = na
                #print(na)

            # if summary.find('strong', text='Frost resistant:'):
            #     frfr = summary.find('strong', text='Frost resistant:').next_element.next_element.next_element
            #     print(fr + frfr + "\n")
            # else:
            #     print(fr + na + "\n")

            '''Rock Specifications'''
            if summary.find('strong', text='Bulk density:'):
                bulk_value = summary.find('strong', text='Bulk density:').parent.next_sibling.text
                #print(bulk_value)
            else:
                bulk_value = na
                #print(na)

            if summary.find('strong', text='Pressure resistance:'):
                pres_value = summary.find('strong', text='Pressure resistance:').parent.next_sibling.text
                #print(pres_value)
            else:
                pres_value = na
                #print(na)

            if summary.find('strong', text='Porosity:'):
                poro_value = summary.find('strong', text='Porosity:').parent.next_sibling.text
                #print(poro_value)
            else:
                poro_value = na
                #print(na)

            if summary.find('strong', text='Frost resistant:'):
                fros_value = summary.find('strong', text='Frost resistant:').next_sibling.next_sibling
                #print(fros_value)
            else:
                fros_value = na
                #print(na)

            '''INSK'''
            if summary.find('strong', text='INSK Nummer:'):
                insk1 = summary.find('strong', text='INSK Nummer:').parent.next_sibling.text
                #print(insk1)
            else:
                insk1 = na
                #print(na)

            if summary.find('strong', text='INSK Nummer alt:'):
                insk2 = summary.find('strong', text='INSK Nummer alt:').parent.next_sibling.text
                #print(insk2)
            else:
                insk2 = na
                #print(na)

            convert_json(num, rock_name, rock_class, color, place1, place2, trade, pq, pd, sff, ar, bulk_value,
                         pres_value, poro_value, fros_value, insk1, insk2, na, width, height, file_path, imageURL)

        with open('rocks_db.json', 'w') as outfile:
            json.dump(rocksArr, outfile, indent=2)


rocksArr = []


def convert_json(num, rock_name, rock_class, color, place1, place2, trade, pq, pd, sff, ar, bulk_value, pres_value,
                 poro_value, fros_value, insk1, insk2, na, width, height, file_path, url):
    rocks_object = {
        "ID": num,
        "Classe": rock_class,
        "Diretorio Img": file_path,
        "Nome da Pedra": rock_name,
        "Largura da Imagem": width,
        "Altura da Imagem": height,
        "Cor": color,
        "Regiao": place1.strip(),
        "Pais": place2.strip(),
        "Nome de comercio": trade.strip(),
        "Qualidade Polimento": pq,
        "Durabilidade Polimento": pd,
        "Pavimentos": sff,
        "Resistencia ao acido": ar,
        "Densidade aparente": bulk_value,
        "Resistencia a pressao": pres_value,
        "Porosidade": poro_value,
        "Resistencia ao frio": fros_value,
        "INSK": insk1,
        "INSK alt": insk2,
        "URL": url,
    }
    rocksArr.append(rocks_object)




get_rock_details()
