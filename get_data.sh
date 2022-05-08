#!/usr/bin/sh

wget "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip" -O DIV2K_LR.zip
wget "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" -O DIV2K_HR.zip
wget "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip" -O DIV2K_val_bicubic.zip
wget "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" -O DIV2K_val_HR.zip
unzip DIV2K_LR.zip -d div2k
unzip DIV2K_HR.zip -d div2k
unzip DIV2K_val_bicubic.zip -d div2k
unzip DIV2K_val_HR.zip -d div2k
rm DIV2K_LR.zip
rm DIV2K_HR.zip
rm DIV2K_val_bicubic.zip
rm DIV2K_val_HR.zip
