import cv2 as cv
import numpy as np
import os

'''Denne kode er genbrugt fra P0-projektet på DAKI.'''

# Main function containing the backbone of the program
def main():
    # foldere til billeder defineres
    output_folder = "/Users/dannihedegaarddahl/Documents/GitHub/P0"
    input_folder = "/Users/dannihedegaarddahl/Documents/GitHub/P0/King Domino testset"

    # vi kigger igennem hver fil i input folderen
    for filename in os.listdir(input_folder):
        # da billederne er navngivet som 1.jpg, 2.jpg osv. 
        # bruger vi filename counteren til at definere billedet vi vil have
        image_path = input_folder +"/" +filename
        # vi udskriver stien til billedet som man kan kigge på hvis der skulle opstå fejl
        print(image_path)
        # her tjekker vi om billedet findes i mappen og skriver "Image not found" hvis det ikke eksisterer
        if not os.path.isfile(image_path):
            print("Image not found")
            return
        # her bruger vi openCV til at åbne billedet
        image = cv.imread(image_path)
        # her bruger vi get_tiles funktionen (se kingdomino.py linje 34) 
        # med billedet som parameter til at få vores tiles
        tiles = get_tiles(image)
        # her udskriver vi mængden af tiles vi har fået, 
        # det kan bruges til fejlfinding hvis der skulle gå noget galt
        print(len(tiles))
        # her kører vi et nested for loop som kører over hver kolonne i hvert række af vores tiles
        # grunden til at vi gør det er fordi get_tiles funktionen giver et to-dimensionelt array ud som ser sådan ud:
        # [[tile1 ],[tile2 ],[tile3 ],[tile4 ],[tile5 ],
        #  [tile6 ],[tile7 ],[tile8 ],[tile9 ],[tile10],
        #  [tile11],[tile12],[tile13],[tile14],[tile15],
        #  [tile16],[tile17],[tile18],[tile19],[tile20],
        #  [tile21],[tile22],[tile23],[tile24],[tile25]]
        # så vi har et 5 gange 5 array som repræsenterer de tiles vi har fra vores spilleplader
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                save_tile(tile, output_folder, filename, x, y)

# her definerer vi en funktion 'save_tile' med inputparametrene tile, outputfolder, image_name, x og y
def save_tile(tile, output_folder, image_name, x, y):
    # her laver vi en mappe 'blandet' hvis den ikke findes
    if not os.path.exists(os.path.join(output_folder, "blandet")):
        os.makedirs(os.path.join(output_folder, "blandet"))

    # her definerer vi navnet på det tile der skal gemmes som f.eks. 1_3_2.png ved brug af en f-streng
    tile_filename = f"{image_name}_{x}_{y}.png"

    # her definerer bi tile_path som er stedet hvor vi vil gemme vores tile
    tile_path = os.path.join(output_folder, "blandet", tile_filename)
    # her gemmer vi vores tile som tile_filename i folderen 'blandet'
    cv.imwrite(tile_path, tile)

    # her skriver vi til konsollen at vi har gemt vores til i 'blandet' folderen
    print(f"Saved Tile as {tile_filename} in 'blandet' folder")

def get_tiles(image):
    # laver en tom liste 
    tiles = []
    # kører et for loop hvor elementer vil blive tilføjet til listen, hvor y repræsenterer en række af billedet
    for y in range(5):
        tiles.append([])
    # kører et nested loop, hvor billedet bliver delt op i en tavel med 25 kvadrater af 100,100 px og tilføjer til listen tiles
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles
# her kører vi main() funktionen
if __name__ == "__main__":
    main()