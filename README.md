# CoastSnap Community Beach Monitoring ToolBox

This toolbox was written by Dr. Mitchell Harley at the University of New South Wales (Australia). Please refer to the [publication in Coastal Engineering](https://www.sciencedirect.com/science/article/abs/pii/S0378383918304551) for further information.

## Getting Started

1.  [Download the CoastSnap Starters Toolkit](https://unsw-my.sharepoint.com/:u:/g/personal/z2273773_ad_unsw_edu_au/EdgaW1h8u0NPh0u8lM1opi4Bb8gVSzVGAhtP4-Eg5jeNEQ?e=rumGKJ) and unzip contents to your local directory. This directory will be your BASE_PATH (needed for step 3)

2.  Download the MATLAB files of this GitHub site to the CoastSnap/Code directory

3.  Make sure you have the following Matlab toolkits installed: mapping toolbox, Statistics and ML toolbox, Image processing toolbox and Curve Fitting Toolbox

4.  Update the following two files to match your local setup: CSPsetPaths.m and CSPloadPaths.m

Thats all - you should now be up and running

For an indepth tutorial refer to the following video: https://www.youtube.com/watch?v=Al7vq26dlyk

## Demo using manly CoastSnap example

![](demo.gif)

## Setting up a new CoastSnap station

Setting up a new CoastSnap station requires three steps:

1.  In the **Images** directory, create a copy of **create_newsitename_here** and rename it to your sitename (e.g. *nthnarrabn*, *manly*).

2.  In the **Shorelines** directory, create a copy of **create_newsitename_here** and rename it to your sitename (e.g. *nthnarrabn*, *manly*).

3.  In the **CoastSnapDB.xlsx** (found in the **Database** folder), create a new tab with the *exact* name as your sitename above. It is best to copy an existing tab like **manly** as the structure of this needs to be identical.

## Inserting relevant station metadata

1.	Open DB spreadsheet (CoastSnap\Database\CoastSnapDB.xlsx)
2.	Copy a new sheet with the name of the new site (same as the name in the Images folder)
3.	Copy in the data from your RTK-GNSS survey of the CoastSnap cradle in B2:B4
4.	
 ![Picture1](https://github.com/user-attachments/assets/f6ea2a12-4fbe-4b29-b834-0cb066110d04)

6.	Approximate the X- and Y-limits for the rectification area. These define the area for rectification where the CoastSnap station is the at the origin X, Y = 0
Ensure you capture enough of the water so that the shoreline detection algorithmn functions correctly! This can be estimated in QGIS. Example below is measured anti-clockwise and results in (approximate):
a.	Xlimit left = -200
b.	Xlimit right = 130
c.	Ylimit lower = 0
d.	Ylimit upper = 600

![Picture2](https://github.com/user-attachments/assets/e1a5bf36-2712-428b-af8f-ee308d8c16c9)

7.	Set the Azimuth Estimate (angle from north). This can be estimated in QGIS

 ![Picture3](https://github.com/user-attachments/assets/b0ad2015-0bad-46b0-8e5b-262e33632431)
 
8.	Set the Initial Tilt (tilt of the photo). This is essentially 90o minus the angle of the cradle.
9.	Set the Tide file name. You may need to create a new tide file if there isn’t an existing appropriate one for the new site.
10.	Set the Characteristic beach slope (under Shoreline Mapping Settings). This can be based on an in-situ RTK-GNSS survey of the intertidal beach slope or estimated from Coastsat (http://coastsat.wrl.unsw.edu.au/) 
11.	Estimate the Tidal Offset using the setup calculator tool in the spreadsheet
12.	Enter the Name, Easting, Northing and Elevation of the measured GCPs under Ground Control Points.
13.	Enter a GCP rectification combo that includes all the GCP surveys (first GCP is 1). For example, for 13 GCPs enter [1:13].


## Managing images using the CoastSnap Database

All new images from various the sources (e.g. Instagram, Facebook, Email) are to be saved to the **Raw** folder for each respective station.





## CoastSnap GUI


