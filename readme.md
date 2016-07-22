# nAud: music information retrieval using deep neural networks
For information on the research side of what was done, see [the Github Pages](http://grey280.github.io/nAud/) section.

## Initial setup
The files expect several things. Most important is the [Keras library](http://keras.io), which you can install via `pip`.

Next, make the expected directories - both in the root level of the repository. One named `output`, for the saved neural networks, and one named `cache`, which will be used to store the library. (You may wish to use a symlink to create the `cache` folder on an external drive with sufficient free space, if your working drive has limited storage.)

Finally, download [ffmpeg](https://ffmpeg.org) and put the executable file in the root of the repository.

## Genre Identification
To run the genre identification network, you'll first need to build the database. Put the local computer's iTunes library file (~/Music/iTunes/iTunes Music Library.xml or C:\Users\{username}\My Music\iTunes\iTunes Music Library.xml) in the 'data' folder of the repository, and then put its filename in `genreid_buildLib.py` or `genreid_buildLib_limited.py`. (Using the 'limited' option is recommended, though you'll need to change the number of songs of each genre to use.)

Run whichever variant of `buildLib` you opted to use - it'll take a while to run. From there, you can input the name of the database file it generates (in the `cache` folder) into `genreid_train.py` and run your code.

To evaluate, make sure `genreid_evaluate.py` has the name of the correct library file, and then run it with the output being stored to disc. It can then be read into the spreadsheet software of your choice - the data is separated by the '/' character, though this *will* yield an extra column full of 'cache', as the output includes the full location of each file.

## Instrument Identification
