
clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      Cáptura imágenes
%%%%%%%%%%%%
% Verifica las cámaras conectadas
info = imaqhwinfo('winvideo'); 
numCameras = length(info.DeviceInfo);

if numCameras < 2
    error('Se necesitan al menos dos cámaras conectadas.');
end
    
% Inicializa las cámaras 
cam1 = videoinput('winvideo', 1, 'YUY2_320x240'); 
cam2 = videoinput('winvideo', 2, 'YUY2_320x240'); 

% Configura las cámaras para captura continua
set(cam1, 'FramesPerTrigger', Inf);
set(cam2, 'FramesPerTrigger', Inf);

% Abre una figura para mostrar la transmisión en vivo
figure;
hImage1 = subplot(1, 2, 1); % Para mostrar la cámara 1
hImage2 = subplot(1, 2, 2); % Para mostrar la cámara 2
title(hImage1, 'Cámara 1');
title(hImage2, 'Cámara 2');

% Inicia la captura continua
start(cam1);
start(cam2);

% Muestra la transmisión en vivo hasta que el usuario presione 'Enter'
disp('Presiona "Enter" para capturar una foto o "q" para salir.');
capturaHecha = false;

while ~capturaHecha
    try
        % Obtén imágenes de cada cámara
        img1_YUY2 = getsnapshot(cam1);
        img2_YUY2 = getsnapshot(cam2);

        % Convierte las imágenes de YUY2 a RGB
        img1_RGB = ycbcr2rgb(img1_YUY2);
        img2_RGB = ycbcr2rgb(img2_YUY2);
        
        % Actualiza las imágenes en la figura sin guardar en memoria
        imshow(img1_RGB, 'Parent', hImage1);
        imshow(img2_RGB, 'Parent', hImage2);
    catch
        disp('Error al obtener la imagen de una de las cámaras.');
        break;
    end

    pause(0.1); % Pequeña pausa para reducir carga de CPU
    k = get(gcf, 'CurrentCharacter');
    
    % Toma la foto si se presiona 'Enter'
    if k == 13  % Código ASCII para 'Enter'
        capturaHecha = true;
        disp('Foto capturada y guardada.');
        
        % Guarda inmediatamente cada captura en el disco
        imwrite(img1_RGB, 'C:\Users\Ana Pau\Desktop\Mice image\captura_cam1.jpg');
        imwrite(img2_RGB, 'C:\Users\Ana Pau\Desktop\Mice image\captura_cam2.jpg');
        disp('Las imágenes se han guardado como captura_cam1.jpg y captura_cam2.jpg.');
    
    % Sal si el usuario presiona 'q'
    elseif k == 'q'
        disp('Captura cancelada por el usuario.');
        break;
    end
end

% Detiene y limpia las cámaras
stop(cam1);
stop(cam2);
delete(cam1);
delete(cam2);
clear cam1 cam2;
disp('Cámaras cerradas correctamente.');

% mostrar imagen, repetir si no 
%%
%%%%%%%%%%%%%%%%%%%%%%%%        Nose Detection 
%% Nose Detection
% Leer la imagen y definir región de interés
upperView = imread('C:\Users\Ana Pau\Desktop\Mice image\captura_cam1.jpg');



% Mostrar imagen original
figure; imshow(upperView); title('Original Image');

[imgHeight, imgWidth, ~] = size(upperView);

% Crop image to only where the ball is visible (assuming the mice will always be on top of the ball)
cropLeft = 60;
cropRight = imgWidth - 80;
cropTop = 40;
croppedImg = upperView(:, cropLeft:cropRight, :);

% Convert to gray scale
grayImg = rgb2gray(croppedImg);


% Paso 1: Detección de objetos circulares con un umbral de brillo
threshold = 150; 
filteredImg = grayImg < threshold;



% Morphological operations
sizeThreshold = 8;
cleanedImg = bwareaopen(filteredImg, sizeThreshold); % Remueve objetos pequeños
closedImg = imclose(cleanedImg, strel('disk', 10));


% Paso 3: Filtro de objetos con valores de rojo > 150
redChannel = croppedImg(:, :, 1);
RedEroded = redChannel < 170;


% Detectar bordes usando el filtro de Canny
edges = edge(RedEroded, 'canny');

% Aplicar transformada de Hough para encontrar líneas
[H, theta, rho] = hough(edges);

% Detectar picos en la transformada de Hough
peaks = houghpeaks(H, 20, 'Threshold', 0.3 * max(H(:)));

% Encontrar líneas usando la función houghlines
lines = houghlines(edges, theta, rho, peaks, 'FillGap', 5, 'MinLength', 10);

% Buscar todas las líneas horizontales mayores a x píxeles
horizontalLinesY = [];
for i = 1:length(lines)
    lineLength = norm(lines(i).point1 - lines(i).point2);
    if abs(lines(i).point1(2) - lines(i).point2(2)) < 5 && lineLength > 10
        horizontalLinesY = [horizontalLinesY; lines(i).point1(2)];
    end
end

% Ordenar posiciones Y de las líneas horizontales
horizontalLinesY = unique(sort(horizontalLinesY));

% Mostrar líneas horizontales detectadas
figure; imshow(upperView); hold on;
title('Horizontal Lines');
for y = horizontalLinesY'
    line([1, imgWidth], [y, y], 'Color', 'red', 'LineWidth', 1.5);
end


% Si no se detectan líneas horizontales
if isempty(horizontalLinesY)
    disp('No horizontal lines detected. Proceeding without cropping sections.');
    % Add a fallback line to simulate one region spanning the image height
    horizontalLinesY = [1; imgHeight];
end

% Recorte entre las líneas horizontales y detección de centróides
for j = 1:length(horizontalLinesY)-1
    croppedSection = closedImg(horizontalLinesY(j):horizontalLinesY(j+1), :);
    figure; imshow(croppedSection); title(sprintf('Cropped Section %d', j));
    
    % Calcular propiedades de las regiones
    [labeledObjects, numObjects] = bwlabel(croppedSection);
    props = regionprops(labeledObjects, 'Area', 'Centroid');
    validProps = props([props.Area] >= 1 & [props.Area] <= 200);

    % Mostrar centróides detectados
    figure; imshow(upperView); hold on;
    title('Centroids Detected');
    for k = 1:length(validProps)
        centroid = validProps(k).Centroid + [cropLeft, horizontalLinesY(j)];
        plot(centroid(1), centroid(2), 'r*', 'LineWidth', 2);
    end
end

% Inicializar variables
centroidMatrix = [];
topRedCentroid = [];
Nose = [];

for j = 1:length(horizontalLinesY)-1
    % Sección recortada
    croppedSection = closedImg(horizontalLinesY(j):horizontalLinesY(j+1), :);
    
    % Mostrar las imágenes de cada corte en la misma figura (subplots)
    figure;
    subplot(1, 2, 1); % Primer subplot: imagen cerrada
    imshow(croppedSection);
    title(sprintf('Cropped Section %d', j));

    % Calcular propiedades de las regiones en la imagen cerrada
    [labeledObjects, numObjects] = bwlabel(croppedSection);
    props = regionprops(labeledObjects, 'Area', 'Centroid');
    validProps = props([props.Area] >= 1 & [props.Area] <= 200);

    % Mostrar los centróides de la imagen cerrada
    subplot(1, 2, 2); % Segundo subplot: centróides detectados
    imshow(upperView); hold on;
    title('Centroids Detected');
    for k = 1:length(validProps)
        centroid = validProps(k).Centroid + [cropLeft, horizontalLinesY(j)];
        plot(centroid(1), centroid(2), 'r*', 'LineWidth', 2);
    end

    % Sección roja
    RedSection = RedEroded(horizontalLinesY(j):horizontalLinesY(j+1), :);
    
    % Calcular propiedades de las regiones rojas
    [labeledRedObjects, numRedObjects] = bwlabel(RedEroded);
    props = regionprops(labeledRedObjects, 'Area', 'Centroid');
    validRedProps = props([props.Area] >= 2 & [props.Area] <= 10);

    redCentroids = [];
    for k = 1:length(validRedProps)
        redCentroids = [redCentroids; validRedProps(k).Centroid + [cropLeft, horizontalLinesY(j)]];
    end

    if isempty(redCentroids)
        disp(['No centroids detected in section ' num2str(j) '.']);
        continue;
    end

    % Calcular el centróide superior de las regiones rojas
    [~, topIdx] = min(redCentroids(:, 2));
    topRedCentroid = redCentroids(topIdx, :);

    % Verificar si un centróide rojo está entre los valores X de los centróides cerrados
    for m = 1:length(validProps)-1
        closedCentroid1 = validProps(m).Centroid + [cropLeft, horizontalLinesY(j)];
        closedCentroid2 = validProps(m+1).Centroid + [cropLeft, horizontalLinesY(j)];

        % Verificar si los centróides rojos están entre los valores X de los de la imagen cerrada
        for k = 1:size(redCentroids, 1)
            redCentroid = redCentroids(k, :);
            if redCentroid(1) > min(closedCentroid1(1)) && ...
               redCentroid(1) < max(closedCentroid2(1)) 
                Nose = redCentroid;
                disp('Nose detected.');
                disp(Nose);
                imwrite(croppedSection, 'C:\Users\Ana Pau\Desktop\Mice image\nose_cam1.jpg');
                break;
            end
        end
        
        % Si se detectó el "Nose", salir del bucle
        if ~isempty(Nose)
            break;
        end
    end

    % Visualizar los resultados en la misma figura
    figure; 
    subplot(1, 2, 1); % Imagen con centróides
    imshow(upperView); hold on;
    if ~isempty(Nose)
        plot(Nose(1), Nose(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
    end
    title('Final Centroids');

    subplot(1, 2, 2); % Imagen final de la sección roja
    imshow(RedSection); hold on;
    title('Red Section');
end

disp('Centroid detection complete.');

%%
%%%%%%%%%%%%%%%%%%%%%            Eye Detection

% Read the image
sideView = imread('C:\Users\Ana Pau\Desktop\Mice image\captura_cam2.jpg'); 

% Convert to grayscale to simplify detection of dark regions
grayImage = rgb2gray(sideView);

% Threshold to detect nearly black areas
darkThreshold = 110; % Adjust based on your specific image
darkMask = grayImage < darkThreshold;

% Perform morphological operations to clean up the binary image
%darkMask = imopen(darkMask, strel('disk', 7)); % Remove small noise
darkMask = imopen(darkMask, strel('disk', 1)); % Remove small noise
darkMask = imclose(darkMask, strel('disk', 7)); % Close gaps in larger objects
figure(3);
imshow (darkMask);

% Label connected components in the binary image
[labeledImage, numObjects] = bwlabel(darkMask);

% Measure properties of labeled regions
stats = regionprops(labeledImage, 'Centroid', 'Area', 'Perimeter', 'Circularity');

% Initialize an array to hold the centroids of circular objects
circularCentroids = [];

% Define size range for circles (in pixels)
%minCircleArea = 30; % Minimum area of circle -> the closer to the mice , the larger the minCircle area 
minCircleArea = 19; % Minimum area of circle -> the closer to the mice , the larger the minCircle area 
maxCircleArea = 30; % Maximum area of circle

% Loop over each object to filter by circularity and size
for k = 1:numObjects
    % Get properties of the region
    circularity = stats(k).Circularity; % Circularity metric (1 = perfect circle)
    area = stats(k).Area;
    centroid = stats(k).Centroid;
    
    % Filter based on circularity, area, and other properties
    if circularity > 0.5 && area >= minCircleArea && area <= maxCircleArea
        circularCentroids = [circularCentroids; centroid]; % Store centroid of valid circular object
    end
end

% Display the original image with centroids of valid circular objects overlaid
figure(4);
imshow(sideView);
hold on;

% Overlay the centroids of detected circular objects
for i = 1:size(circularCentroids, 1)
    plot(circularCentroids(i, 1), circularCentroids(i, 2), 'r*', 'MarkerSize', 10);
end

% Title and labels
title('Detected Circular Objects of Specific Size and Centroids');
xlabel('X-coordinate');
ylabel('Y-coordinate');


%%
%%%%%%%%%%%%%%%%%%%%%             Coordenates

% Define goal coordinates in pixels
goalNose = [155, 100]; % Nose target
goalEye = [175, 150];  % Eye target
Eye = circularCentroids;

% Load camera calibration parameters
load('LogiParams.mat'); % Ensure you are using the correct file

% Create a new figure
figure;

% Plot the upper view image with points and arrows
subplot(1, 2, 1); % Divide the figure into 1 row and 2 columns, select first plot
imshow(upperView);
hold on;

% Mark points on the image
plot(Nose(1), Nose(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Nose');
plot(goalNose(1), goalNose(2), 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Goal Nose');

% Draw arrow between points
quiver(Nose(1), Nose(2), goalNose(1) - Nose(1), goalNose(2) - Nose(2), 0, ...
       'MaxHeadSize', 0.5, 'Color', 'g', 'LineWidth', 2, 'DisplayName', 'Difference');
title('Upper View');
legend('show');
hold off;

% Plot the side view image with points and arrows
subplot(1, 2, 2); % Select the second plot
imshow(sideView);
hold on;

% Mark points on the image
plot(Eye(1), Eye(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Eye');
plot(goalEye(1), goalEye(2), 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Goal Eye');

% Draw arrow between points
quiver(Eye(1), Eye(2), goalEye(1) - Eye(1), goalEye(2) - Eye(2), 0, ...
       'MaxHeadSize', 0.5, 'Color', 'g', 'LineWidth', 2, 'DisplayName', 'Difference');
title('Side View');
legend('show');
hold off;

% Extract camera intrinsics and world point transformations
intrinsics = LogiParams.Intrinsics;
R = LogiParams.RotationMatrices(:, :, 1); % First calibrated view
t = LogiParams.TranslationVectors(1, :);

% Convert image points to world coordinates
worldPointsNose = pointsToWorld(LogiParams, R, t, Nose);
worldPointsGoalNose = pointsToWorld(LogiParams, R, t, goalNose);
worldPointsEye = pointsToWorld(LogiParams, R, t, Eye);

% Calculate differences in mm
deltaX = worldPointsGoalNose(1) - worldPointsNose(1); % Y in world coordinates
deltaY = worldPointsGoalNose(2) - worldPointsNose(2); % X in world coordinates
deltaZ = worldPointsGoalNose(2) - worldPointsEye(2); % Z for Eye comparison

% Calculate total distance in mm
distance = sqrt(deltaX^2 + deltaY^2);

% Print results to console
fprintf('Difference in X: %.3f mm\n', deltaX);
fprintf('Difference in Y: %.3f mm\n', deltaY);
fprintf('Difference in Z: %.3f mm\n', deltaZ);
fprintf('Total Distance: %.3f mm\n', distance);

% Wait for user input
disp('Press the space bar to continue, or any other key to restart.');
key = waitforbuttonpress;
if key == 1 && strcmp(get(gcf, 'CurrentCharacter'), 'c') % "c" key pressed
    disp('Sending command to motors');
else
    disp('Restarting the script...');
    run('NoseDetectionClean.m'); % Restart the script
    return;
end

%%  
%%%%%%%%%%%%%%%%%%%%              Motors

% micro stepping to 320 in UGS
try
    % Define variables for movement
 
    deltaZ = -18; % 
    feedRate = 50; % Feed rate in mm/min

    % Create serial port object for communication
    s = serialport('COM12', 115200); % Replace 'COM12' with your device's port number

    % Set units to millimeters and choose positioning mode
    writeline(s, 'G21'); % Set units to millimeters
    writeline(s, 'G91'); % Relative positioning mode
    pause(0.1); % Small delay
    disp(['Initial setup feedback: ', readline(s)]);

    % Format G-code command with the variables
    gcodeCommand = sprintf('G1 X%.2f Y%.2f Z%.2f F%.2f', deltaX, deltaY, deltaZ, feedRate);

    % Send the formatted G-code command
    writeline(s, gcodeCommand);
    pause(5); % Wait for movement to complete
    disp(['Move command feedback: ', readline(s)]);

    % Close the serial port connection
    clear s;
catch err
    disp('Error encountered:');
    disp(err.message);
    % Close serial port if open
    if exist('s', 'var')
        clear s;
    end
end

