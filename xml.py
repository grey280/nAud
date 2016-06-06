from xml.dom.minidom import parse
import xml.dom.minidom

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse("data/iTunesLibrary.xml")
collection = DOMTree.documentElement
if collection.hasAttribute("tracks"):
   print ("Root element : {}".format(collection.getAttribute("tracks")))

# Get all the movies in the collection
movies = collection.getElementsByTagName("track")

# Print detail of each movie.
for track in tracks:
   print("*****Track*****")
   if track.hasAttribute("title"):
      print("Title: {}".format(track.getAttribute("title")))

   # type = movie.getElementsByTagName('type')[0]
   # print("Type: %s" % type.childNodes[0].data)
   # format = movie.getElementsByTagName('format')[0]
   # print "Format: %s" % format.childNodes[0].data
   # rating = movie.getElementsByTagName('rating')[0]
   # print "Rating: %s" % rating.childNodes[0].data
   # description = movie.getElementsByTagName('description')[0]
   # print "Description: %s" % description.childNodes[0].data