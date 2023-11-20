#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/ram.h"

// Spi Master configuration
#define SPI_MASTER_BAUDRATE             1000000
#define SPI_MASTER_WORDSIZE             PI_SPI_WORDSIZE_8
#define SPI_MASTER_ENDIANESS            1
#define SPI_MASTER_POLARITY             PI_SPI_POLARITY_0
#define SPI_MASTER_PHASE                PI_SPI_PHASE_0
#define SPI_MASTER_CS                   1                                   // CSO (GPIO34)
#define SPI_MASTER_ITF                  1                                   // SPI1 peripheral is set as master
#define SPI_MASTER_DUMMY_CYCLE          3
#define SPI_MASTER_DUMMY_CYCLE_MODE     PI_SPI_DUMMY_CLK_CYCLE_BEFORE_CS

// Spi Master pad configuration
#define SPI_MASTER_PAD_SCK              PI_PAD_033  // Ball B2          > CN2 connector, pin 5
#define SPI_MASTER_PAD_CS0              PI_PAD_034  // Ball E4          > CN2 connector, pin 4
#define SPI_MASTER_PAD_SDO              PI_PAD_038  // Ball A1  (MOSI)  > CN2 connector, pin 1
#define SPI_MASTER_PAD_SDI              PI_PAD_039  // Ball E3  (MISO)  > CN3 connector, pin 5
#define SPI_MASTER_PAD_CS1              PI_PAD_067  // Ball E4          > CN2 connector, pin 4
#define SPI_MASTER_PAD_FUNC             PI_PAD_FUNC0
#define SPI_MASTER_IS_SLAVE             0

// SPI Slave Configuration
#define SPI_SLAVE_BAUDRATE              150000000
#define SPI_SLAVE_WORDSIZE              PI_SPI_WORDSIZE_8
#define SPI_SLAVE_ENDIANESS             1
#define SPI_SLAVE_POLARITY              PI_SPI_POLARITY_0
#define SPI_SLAVE_PHASE                 PI_SPI_PHASE_0
#define SPI_SLAVE_CS                    0                   // (GPIO57) Use SPI2 CS0 on GAP9EVK's CN6(7) connector
#define SPI_SLAVE_ITF                   2                   // SPI2 peripheral is set as slave
#define SPI_SLAVE_IS_SLAVE              1

// Spi slave pad configuration
#define SPI_SLAVE_PAD_SCK               PI_PAD_056  // Ball H1  > GAP9 EVK CN6 connector, pin 3
#define SPI_SLAVE_PAD_CS0               PI_PAD_057  // Ball H2  > GAP9 EVK CN6 connector, pin 7
#define SPI_SLAVE_PAD_CS1               PI_PAD_053  // Ball H7  > GAP9 EVK CN5 connector, pin 10
#define SPI_SLAVE_PAD_SDO               PI_PAD_059  // Ball H4  > GAP9 EVK CN6 connector, pin 2
#define SPI_SLAVE_PAD_SDI               PI_PAD_058  // Ball H3  > GAP9 EVK CN6 connector, pin 6
#define SPI_SLAVE_PAD_FUNC              PI_PAD_FUNC2

#define SPI_NO_OPTION                   0 // No option applied for SPI transfer


#define MAGIC_NUMBER_H    0x47
#define MAGIC_NUMBER_L    0x41
#define CMD_STATUS_GET    0xf1
#define CMD_STATUS_SEND   0xf2
#define STATUS_RDY        0xD1


void spi_slave_init(pi_device_t* spi_slave, struct pi_spi_conf* spi_slave_conf);

void send_image_spi(pi_device_t* spi_slave,uint8_t*img, uint16_t img_w, uint16_t img_h);

void send_jpeg_spi(pi_device_t* spi_slave,  uint8_t* img, int img_size, unsigned int *perf_array);

void send_image_spi_ram(pi_device_t* spi_slave,pi_device_t* ram,uint32_t img, uint16_t img_w, uint16_t img_h);