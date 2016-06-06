from xml.dom.minidom import parse
import xml.dom.minidom

DOMTree = xml.dom.minidom.parse("data/iTunesLibrary.xml")
collection = DOMTree.documentElement
print(collection)

tracks = 