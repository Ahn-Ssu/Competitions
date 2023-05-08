features=(64,128,256,512,1024,64)
BasicUNet(
  (conv_0): TwoConv(
    (conv_0): Convolution(
      (conv): Conv3d(4, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (adn): ADN(
        (D): Dropout(p=0.0, inplace=False)
        (A): ReLU()
      )
    )
    (conv_1): Convolution(
      (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (adn): ADN(
        (D): Dropout(p=0.0, inplace=False)
        (A): ReLU()
      )
    )
  )
  (down_1): Down(
    (max_pooling): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (convs): TwoConv(
      (conv_0): Convolution(
        (conv): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
      (conv_1): Convolution(
        (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
    )
  )
  (down_2): Down(
    (max_pooling): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (convs): TwoConv(
      (conv_0): Convolution(
        (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
      (conv_1): Convolution(
        (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
    )
  )
  (down_3): Down(
    (max_pooling): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (convs): TwoConv(
      (conv_0): Convolution(
        (conv): Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
      (conv_1): Convolution(
        (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
    )
  )
  (down_4): Down(
    (max_pooling): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (convs): TwoConv(
      (conv_0): Convolution(
        (conv): Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
      (conv_1): Convolution(
        (conv): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
    )
  )
  (upcat_4): UpCat(
    (upsample): UpSample(
      (deconv): ConvTranspose3d(1024, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    )
    (convs): TwoConv(
      (conv_0): Convolution(
        (conv): Conv3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
      (conv_1): Convolution(
        (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
    )
  )
  (upcat_3): UpCat(
    (upsample): UpSample(
      (deconv): ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    )
    (convs): TwoConv(
      (conv_0): Convolution(
        (conv): Conv3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
      (conv_1): Convolution(
        (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
    )
  )
  (upcat_2): UpCat(
    (upsample): UpSample(
      (deconv): ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    )
    (convs): TwoConv(
      (conv_0): Convolution(
        (conv): Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
      (conv_1): Convolution(
        (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
    )
  )
  (upcat_1): UpCat(
    (upsample): UpSample(
      (deconv): ConvTranspose3d(128, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    )
    (convs): TwoConv(
      (conv_0): Convolution(
        (conv): Conv3d(192, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
      (conv_1): Convolution(
        (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (adn): ADN(
          (D): Dropout(p=0.0, inplace=False)
          (A): ReLU()
        )
      )
    )
  )
  (final_conv): Conv3d(64, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
