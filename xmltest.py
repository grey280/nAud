import xml.etree.ElementTree as etree

tree = etree.parse('data/iTunesLibrary.xml')

root = tree.getroot()

print(root[0][15])

# tracks = root[0][15.findall()
# print(tracks)



# for child in root:
# 	for c2 in child:
# 		print(c2)
# # 		if c2.tag == "Tracks":
# # 			trackRoot = c2.value

# # print(trackRoot)