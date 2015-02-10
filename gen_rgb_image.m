function [image, partIds] = gen_rgb_image(rgbd, shape)%

image = [];
if rgbd.nRows > 0 && rgbd.nCols > 0
    colorR = uint8(floor(double(rgbd.pointColors)/256/256));
    res = double(rgbd.pointColors) - double(colorR)*256*256;
    colorG = uint8(floor(res/256));
    res = res - double(colorG)*256;
    colorB = uint8(res);
    imgR = uint8(255 - zeros(rgbd.nRows, rgbd.nCols));
    imgG = imgR;
    imgB = imgR;
    imgR(rgbd.pointPixelIds) = colorR;
    imgG(rgbd.pointPixelIds) = colorG;
    imgB(rgbd.pointPixelIds) = colorB;
    image(:,:,1) = imgR;
    image(:,:,2) = imgG;
    image(:,:,3) = imgB;
    image = uint8(image);
    partIds = uint32(zeros(rgbd.nRows, rgbd.nCols));
    if exist('shape','var')
        pointPartIds = shape.partIds(rgbd.pointFaceIds);
        partIds(rgbd.pointPixelIds) = pointPartIds;
    end;
 end