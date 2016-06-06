import xml.etree.ElementTree as etree

tree = etree.parse('data/iTunesLibrary.xml')

root = tree.getroot()

print(root)