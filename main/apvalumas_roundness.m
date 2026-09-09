function metric = apvalumas_roundness(Im)
% Return the circularity (roundness) of the largest detected object.
% Circularity = 4*pi*area/perimeter^2; a perfect circle is close to 1.

BW = im2bw(rgb2gray(Im), 0.95);
BW = imfill(~BW, 'holes');
BW = imopen(BW, strel('disk', 12));

regions = regionprops(double(BW), {'perimeter', 'area'});
if isempty(regions)
    metric = NaN;
    return;
end

[~, largest] = max([regions.Area]);
areaValue = regions(largest).Area;
perimeterValue = regions(largest).Perimeter;

if perimeterValue == 0
    metric = 0;
else
    metric = 4*pi*areaValue/perimeterValue^2;
end
