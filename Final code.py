import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_random_forest(data_path):

    # Reading the dataset as a pandas dataframe with y as target label and X as features. 
    df = pd.read_csv(data_path)
    y = df['Class']
    X = df[['hue_median','saturation_median','value_median','hue_mean','saturation_mean','value_mean']]

    # Splitting dataset into 80% training data and 20% testdata
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialization of the RandomForestClassifier class and training it on the training data.
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

def find_crowns(game_img, templates_path):
    all_rectangles = []
    for template_path in templates_path:
        # Using Template Matching to match crown templates on the board
        crown_template = cv.imread(template_path, cv.IMREAD_UNCHANGED)
        result = cv.matchTemplate(game_img, crown_template, cv.TM_CCOEFF_NORMED)
        threshold = 0.57
        yloc, xloc = np.where(result >= threshold)

        # Loop that saves the x-coord, y-coord, width and height in all_rectangles list
        for i in range(len(xloc)):
            x, y = xloc[i], yloc[i]
            w = crown_template.shape[1]
            h = crown_template.shape[0]
            all_rectangles.append([int(x), int(y), int(w), int(h)])

    rectangles = cv.groupRectangles(np.array(all_rectangles), 1, 0.2)[0]
    return rectangles

def extract_tile_features(tile):
    # Mean and median hue, saturation and value is extracted from the tile. 
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue_median, saturation_median, value_median = np.median(hsv_tile, axis=(0,1))
    hue_mean, saturation_mean, value_mean = np.mean(hsv_tile, axis=(0,1))
    return [hue_median, saturation_median, value_median, hue_mean, saturation_mean, value_mean]

def predict_tile_types(tiles, rf_model):
    # Saving features from each tile using the extract_tile_features function. 
    features = []
    for tile in tiles:
        features.append(extract_tile_features(tile))
    # Dataframe contain features for each tile.
    df_tiles = pd.DataFrame(features, columns=["hue_median", "saturation_median", "value_median", "hue_mean", "saturation_mean", "value_mean"])

    # Using the Random Forest to predict the tile types. 
    return rf_model.predict(df_tiles)

def image_cropper(game_img, rectangles, rf_model, rows=5, cols=5):

    # Creating empty list for tile images and a grid
    tile_images = []
    grid = []

    # Filling the grid with dictionaries containing Type and crown count. 
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append({'Type': None, 'crown_count': 0})
        grid.append(row)
    
    # Calculating the tile height and width using the size of the board
    tile_height = game_img.shape[0] // rows
    tile_width = game_img.shape[1] // cols
    
    # Getting the coordinates of each tile
    for y in range(rows):
        for x in range(cols):
            x_start = x * tile_width
            y_start = y * tile_height
            x_end = (x + 1) * tile_width
            y_end = (y + 1) * tile_height

            tile = game_img[y_start:y_end, x_start:x_end]
            tile_images.append(tile)
    
    # Using predict_tile_types function to get the predicted types 
    predicted_types = predict_tile_types(tile_images, rf_model)
    current_position = 0
    for y in range(rows):
        for x in range(cols):
            grid[y][x]['Type'] = predicted_types[current_position]
            current_position += 1
    
    # Using the crown coordinates to check if crown is in tile. 
    # Adding to crown count in current grid slot if crown is present.
    for y in range(rows):
        for x in range(cols):
            segment_rect = (x * tile_width, y * tile_height, tile_width, tile_height)
            crown_count = 0
            for (crown_x, crown_y, crown_w, crown_h) in rectangles:
                if (crown_x < (segment_rect[0] + segment_rect[2]) and
                    (crown_x + crown_w) > segment_rect[0] and
                    crown_y < (segment_rect[1] + segment_rect[3]) and
                    (crown_y + crown_h) > segment_rect[1]):
                    crown_count += 1
                    grid[y][x]['crown_count'] = crown_count
    return grid

def fill_grid_visited(x, y, grid, counted, rows, cols):
    visited = []
    for _ in range(cols):
        visited.append([False] * rows)
    
    return count_terrain(x, y, grid, visited, counted, rows, cols)

def count_terrain(x, y, grid, visited, counted, rows, cols):
    current_type = grid[x][y]['Type']
    visited[x][y] = True
    counted[x][y] = True  
    count = 1 
    crowns = grid[x][y]['crown_count']

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # Counting connected tiles and adding up the amount of crowns in the connected tiles domain. 
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < cols and 0 <= new_y < rows and not visited[new_x][new_y] and grid[new_x][new_y]['Type'] == current_type:
            counts = count_terrain(new_x, new_y, grid, visited, counted, rows, cols)
            count += counts[0]
            crowns += counts[1]
    return (count, crowns)

def calculate_points(grid, rows, cols):
    counted = []
    for _ in range(cols):
        counted.append([False] * rows)
    # Calling the fill_grid_visited function and calculating the total points 
    # using connected tiles and amount of crowns. 
    sum_of_points = []
    for x in range(rows):
        for y in range(cols):
            if not counted[x][y]:  
                connected_count = fill_grid_visited(x, y, grid, counted, rows, cols)
                points_per_terrain = connected_count[0] * connected_count[1]
                sum_of_points.append(points_per_terrain)
    return sum(sum_of_points)

def analyze_board(board_num, data_path, templates_path, rows=5, cols=5):
    # Function that puts together all functions to the final program. 
    image_path = fr"C:\Users\marti\OneDrive - Aalborg Universitet\DAKI\2. Semester\Design og udvikling af AI-systemer\Miniprojekt\King Domino dataset\Cropped and perspective corrected boards\{board_num}.jpg"
    game_img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    rectangles = find_crowns(game_img, templates_path)
    rf_model = train_random_forest(data_path)
    grid = image_cropper(game_img, rectangles, rf_model, rows, cols)
    total_points = calculate_points(grid, rows, cols)
    print(f"Total points for board number {board_num} is: {total_points}")

def main():
    # Defining paths and calling the overall analyze_board function. Output will be points for each board in the range. 
    data_path = r"C:\Users\marti\OneDrive - Aalborg Universitet\DAKI\2. Semester\Design og udvikling af AI-systemer\Miniprojekt\CSV-filer med features\Alt data\All_data.csv"
    templates_path = [
        r"C:\Users\marti\Desktop\DUAS miniprojekt\Crown0.jpg",
        r"C:\Users\marti\Desktop\DUAS miniprojekt\Crown1.jpg",
        r"C:\Users\marti\Desktop\DUAS miniprojekt\Crown2.jpg",
        r"C:\Users\marti\Desktop\DUAS miniprojekt\Crown3.jpg"]
    for i in range(1, 75):
        analyze_board(i, data_path, templates_path)

if __name__ == "__main__":
    main()
