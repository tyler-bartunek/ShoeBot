# ShoeBot

## Table of Contents
- [Statement of Purpose](#statement-of-purpose)
- [Module Descriptions](#description-of-modules)
- [Hardware](#hardware)
  - [Printed Components and Module Fit](#printed-components-and-module-fit)
  - [Printing Directions](#printing-directions)
  - [Power Distribution and Monitoring](#power-distribution-and-monitoring)
- [Software](#software)
  - [Communication Protocol](#communication-protocol)
  - [Chip Select](#chip-select)
  - [Future development](#future-development)
- [FAQ](#faq)
- [License](#license)

## Statement of Purpose
The ShoeBot framework is an open-source platform designed to make learning about and development of mobile robotic
systems accessible, modular, and affordable. 

A standardized 3D-printable rail mounting system allows for the creation of completely reconfigurable mobile robots,
with pre-designed modules intended to be buildable with nothing more than a 3D printer, screwdriver, and a soldering 
iron. Additional custom hardware files are provided to facilitate wiring, but are optional. 

The guiding philosophy is that if you have a box, soldering iron, basic fasteners, and a 3D printer then you should be 
able to put this system together.

## Module Descriptions

1. Wheels: This module consists of DC motors and mecanum wheels.
2. Shoeshine: Offers quadrapedal motion through use of servomotors
3. HexaBox: Requires six mounts, offers six-legged locomotion through use of servomotors.

Each module has its own Pi Pico in it running in peripheral mode as a local controller, and should have
the following leads coming out of it: 
- 3.3 V (Pico VCC)
- 12 V (Motor Power)
- COPI (SPI)
- CIPO (SPI)
- SCK (SPI)
- CS (SPI)
- Pico GND
- Motor GND

## Hardware

### Printed Components and Module Fit
Both STEP and STL files for all components to be printed are provided under the hardware directory, alongside printing
directions as applicable. This system was prototyped using an Ender 3 V2 and Cura slicer using PLA+, with dimensions set 
to reflect typical tolerances for that printer with the intent of snug clearance fits between mounting rails and modules.
Nominal ridge width for the rail channels is 8 mm, but for the described printer setup a 7.8 mm width with 8.2 mm gap
between ridges was found to provide the desired fit. 

### Printing Directions
All 3D-printed hardware was designed with a specific print orientation in mind to save time and material by reducing the
need for most overhangs. Prior to printing, please consult with the printing directions document as it becomes available 
as it contains images of each part in Cura oriented in the proper direction as well as notes on which holes were designed
for brass inserts to be added for screw mounting. During prototyping, a 20% infill was used.

[Amazon link to brass inserts used during prototyping](https://www.amazon.com/dp/B0DM21Z6XP?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1)

### Power Distribution and Monitoring
This system was developed with a Raspberry Pi 3b+ in mind as the central controller, which desires a pretty stable 5V 2.5A
DC power supply. To achieve this stable supply, the batteries (which will output lower voltage over time as they discharge)
are connected to buck-boost converters. This design opts for a buck converter with micro USB output to leverage the Pi's 
existing polyfuse for a little bit of insurance against instability in the power line. 

The pre-developed modules for this system use 12 V motors and servos. Combining the need for a stable supply for the Pi
with this fact, as well as a desire to leverage bulk pricing discounts, during development two [Pololu S18V20F12](https://www.pololu.com/product/2577)
buck-boosts were used, one dedicated to the motors and the other dedicated to the [buck used for the Pi](https://www.amazon.com/dp/B0B6NZBWV4?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1).

To meet the power needs of the peripheral controllers, which in this iteration of the design are Pi Picos, a simple buck converter 
like the LM2596 is sufficient based on the assumption that properly sized batteries (recommended 3S or 4S lithium polymer) are
selected.

Based on the power needs for the system, a 4A fuse is recommended between the buck-boost converters and the battery. Based
on balance needs, it is recommended to have two batteries installed but only one needs to be connected though this is less
crucial for configurations similar to those achievable for the "Wheels" modules.

## Software

### Communication Protocol
This system presently communicates with locomotion hardware modules over SPI to facilitate both high-speed transactions
for configurations where that matters as well as ease of identifying module configurations and approximate layout. 

#### Chip Select 
In order to free up GPIO pins on the Raspberry Pi, it uses an 8-bit shift register to toggle chip select pins for each of
the modules. The details of how to configure it will depend somewhat on the register you choose to use, but the example
module firmware provided anticipates an active-low chip select so please apply a binary not to whichever value you send to
the register to achieve pulling all other lines high aside from your selection. The provided KiCad design for the SPI
splitting assumes 6 possible module attachment points, with the following addresses (prior to bit inversion):

1. Back Left: 0x80
2. Center Left: 0x40
3. Front Left: 0x20
4. Front Right: 0x10
5. Center Right: 0x08
6. Back Right: 0x04

#### Future development
As time goes on, the controls and motion-planning capabilities of this system will be extended. Check in here for updates.

## FAQ

1. Why shoeboxes?

>I had originally set out to make what essentially amounts to just the wheels module as a CAD and mechatronic portfolio piece.
Seeking to understand the design considerations, I started to immerse myself in literature surrounding mecanum wheel devices 
and discovered something that is in hindsight self-evident: different design decisions like the wheel spacing and orientation 
fundamentally alters the dynamics of the system and therefore influence control. Creating modular systems to explore the impact 
of these parameters was largely underexplored in what I found, so I set out to make my own. But I needed a good frame. Something
cheap and easy to deploy. As I looked around me, I saw an unused shoebox. And the ShoeBot system was born. 

>Realistically any cardboard box will work, but at the time a shoebox seemed like the right size for what I set out to accomplish.

2. Why didn't you use \<insert fastener here\> to fasten the modules to the box?

>In the early days, I explored a number of different means for mounting the hardware to the box. The current design for mounting
hardware to the box solves two problems simultaneously: the first is that of establishing a secure connection with minimal
assembly complexity, the second is offering a path for routing the wires from inside the box to the modules on the outside.

3. Doesn't the act of cutting into the box limit the reusability of a single box to test multiple design parameters?

>Yes, as do other adhesives that have enough strength to reliably anchor the hardware to the outside of the box. 

## License
This project is licensed under the terms of the [MIT License](LICENSE), which permits use, modification, and redistribution 
of the materials in this repository.

If you build upon or share this work, please provide appropriate attribution and link back to this repository so others can 
benefit from and contribute to the project.
