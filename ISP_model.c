/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

#include <stdint.h>
#include <stdio.h>

// AutoTiler Libraries
#include "AutoTilerLib.h"
// Resize generator
#include "ISP_Generators.h"



int main(int argc, char **argv)
{
	BayerOrder_t bayer_BGGR=BGGR;
	//BayerOrder_t bayer_GRBG=GRBG;
	// This will parse AutoTiler options and perform various initializations
	if (TilerParseOptions(argc, argv)) {
		printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
	}
	// Setup AutTiler configuration. Used basic kernel libraries, C names to be used for code generation,
	// compilation options, and amount of shared L1 memory that the AutoTiler can use, here 110000 bytes
	SetInlineMode(ALWAYS_INLINE);
	SetSymbolDynamics();

	SetUsedFilesNames(0, 1, "ISP_BasicKernels.h");
	SetGeneratedFilesNames("ISP_Kernels.c", "ISP_Kernels.h");

    SetMemoryDeviceInfos(3,
        AT_MEM_L1, 110000, "DeMosaic_L1_Memory", 0, 0,
		AT_MEM_L2, 200000, "DeMosaic_L2_Memory", 0, 0,
		AT_MEM_L3_DEFAULTRAM, 0, "DeMosaic_L3_Memory", 0, 1 //This one means that the Ram is externally handled (the user takes care of opening and closing)
    );

	// Load the Resize basic kernels template library
	LoadISPLibrary();
	// Call Resize generator
	unsigned int W = 640, H = 480;

	//GenerateDeMosaic_OutHWC("demosaic_image_HWC", W, H, sizeof(char),0,0,bayer_BGGR);
	//GenerateWB_HWC("white_balance_HWC", W, H, 0);

	Simple_DeMosaic_Resize("demosaic_image_HWC_L3", W,H,0,1,bayer_BGGR);
	//GenerateDeMosaic_OutHWC("demosaic_image_HWC_L3", W, H,sizeof(char), 1,1,bayer_BGGR);
	GenerateWB_HWC("white_balance_HWC_L3", 320,240, 1);


	// Now that we are done with model parsing we generate the code
	GenerateTilingCode();
	return 0;
}
