import xmltodict

data = open("./data/iTunesLibrary.xml", "r")
data = data.read()

dictionary = xmltodict.parse(data)

print(dictionary)