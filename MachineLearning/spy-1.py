import matplotlib.pyplot as plt
sizes=[40,20,25,30]
labels=['python','java','golang','c']
plt.pie(sizes,labels=labels, autopct='%1.1f%%',startangle=90)
plt.title("Languages")
plt.show()
