import matplotlib.pyplot 		as plt
import numpy 					as np
import random

# I like the Tango Desktop Project's color swatch, so I tend to use that
art_color = '#204a87'
pop_color = '#3465a4'
tdt_color = '#729fcf'

# Set up plot
X = np.linspace(0, 1, 32, endpoint=True)
plt.xlabel('Time')
plt.ylabel('Probability')

# Generate fake data
art = np.random.uniform(low=0.0, high=0.5, size=(1,32))
pop_feed = np.random.uniform(low=0.1, high=0.6, size=(1,32))
pop = []
random.shuffle(pop_feed[0])
for num in range(len(pop_feed)):
	pop.append(pop_feed[0][num]+art[num])

art_array = np.asarray(art[0])
pop_array = np.asarray(pop[0])

# Plot fake data
plt.plot(X, art_array, color=art_color)
plt.plot(X, pop_array, color=pop_color)

# Fill the spaces between
plt.fill_between(X, 0, art_array, color=art_color, label="Art")
plt.fill_between(X, art_array, pop_array, color=pop_color, label="Pop")
plt.fill_between(X, pop_array, 1.0, color=tdt_color, label="Traditional")

# Set the size of the plot
axes = plt.gca()
axes.set_ylim([0, 1.0])

# And draw it
plt.legend(loc='upper left')
plt.show()