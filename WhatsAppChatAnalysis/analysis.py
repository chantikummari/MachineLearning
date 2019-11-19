import operator
import matplotlib.pyplot as plt

top_contributors = 5
f = open("whatsappchat.txt")
lines = f.readlines()

###Preparing members list
start = "- "
end = ": "
Memlist = []
for i in lines:
    if "- " in i and ": " in i:
        if i.split(start)[1].split(end)[0] not in Memlist:
            Memlist.append((i.split(start)[1].split(end)[0]))

###Creating the dictionary with all members and messages count.
dict = {}
count = 0
for i in range(len(Memlist)):
    dict[Memlist[i]] = 0
    for line in lines:
        if "- " + list(dict.keys())[i] + ": " in line:
            count +=1
    dict[Memlist[i]] = count
    count = 0

###Top 5 contributors
sorted_dict = sorted(dict.items(),key=operator.itemgetter(1),reverse=True)
print("Top {} contributors in group : {}\n".format(top_contributors,sorted_dict[:top_contributors]))

###List of users with number of messages sent
print("List of users with number of messages sent : ", sorted_dict)

### pie chart generation
others = 0
for i in sorted_dict[top_contributors:]:
    others += i[1]
labels = sorted_dict[0][0],sorted_dict[1][0],sorted_dict[2][0],sorted_dict[3][0],sorted_dict[4][0],'others'
sizes = [sorted_dict[0][1],sorted_dict[1][1],sorted_dict[2][1],sorted_dict[3][1],sorted_dict[4][1], others]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red','orange']
patches, texts,temp  = plt.pie(sizes, colors=colors, shadow=True, startangle=90, autopct='%1.0f%%')
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
