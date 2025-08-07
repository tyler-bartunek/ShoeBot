
# ShoeBot <img src="images/Logo.png" alt="ShoeBot Logo" width="150" align = "right"/>

&nbsp;

## Table of Contents
- [Statement of Purpose](#statement-of-purpose)
- [Quick-Start Guide](#quick-start-guide)
- [Module Descriptions](#description-of-modules)
- [Hardware](#hardware)
  - [Printed Components and Module Fit](#printed-components-and-module-fit)
- [Software](#software)
  - [Communication Protocol](#communication-protocol)
  - [Chip Select](#chip-select)
  - [Future development](#future-development)
- [FAQ](#faq)
- [License](#license)

## Repository Structure
- `/hardware/` – STL and STEP files for 3D printing
- `/software/` – Code for the centralized controller, example firmware for modules, and hardware test scripts
- `/docs/` – Quick-start guide and printing instructions, additional reference material as it becomes available (PDF)
- `/images/` – Holds images used in this README, as well as some others throughout the repository
- `README.md` – You are here.

## Statement of Purpose
The ShoeBot framework is an open-source platform designed to make learning about and development of mobile robotic
systems accessible, modular, and affordable. 

A standardized 3D-printable rail mounting system allows for the creation of completely reconfigurable mobile robots,
with pre-designed modules intended to be buildable with nothing more than a 3D printer, screwdriver, and a soldering 
iron. Additional custom hardware files are provided to facilitate wiring, but are optional. 

The guiding philosophy is that if you have a box, soldering iron, basic fasteners, and a 3D printer then you should be 
able to put this system together.

## Quick-Start Guide
The Quick-Start Guide (coming soon) will walk you through the minimal configuration needed to get the system up and running. 
This includes all required materials, recommended print settings, and step-by-step instructions for assembling and testing 
the basic "Wheels" module.

## Module Descriptions

1. Wheels: This module consists of DC motors and mecanum wheels, represents minimal functional example.
2. Shoeshine: Offers quadrupedal motion through use of servomotors
3. HexaBox: Requires six mounts, offers six-legged locomotion through use of servomotors.

## Hardware

### Printed Components and Module Fit
Both STEP and STL files for all components to be printed are provided under the hardware directory, with directions for
the base hardware and wheels module provided in the Quick-Start Guide. As modules are added, printing directions specific
to those modules will be added to the docs directory. This system was prototyped using an Ender 3 V2 and Cura slicer using 
PLA+, with dimensions set to reflect typical tolerances for that printer with the intent of snug clearance fits between 
mounting rails and modules. Nominal ridge width for the rail channels is 8 mm, but for the described printer setup a 7.8 mm 
width with 8.2 mm gap between ridges was found to provide the desired fit. 

## Software

### Communication Protocol
This system presently communicates with locomotion hardware modules over SPI to facilitate both high-speed transactions
for configurations where that matters as well as ease of identifying module configurations and approximate layout. Included 
as an optional component are the schematic and KiCad board file for a PCB that handles fanning out the SPI communication.

#### Chip Select 
In order to free up GPIO pins on the Raspberry Pi, it uses an 8-bit shift register to toggle chip select pins for each of
the modules. On the optional PCB, this register is a 74HC595, and the following values correspond to the following locations:

1. Back Left: 0x80
2. Center Left: 0x40
3. Front Left: 0x20
4. Front Right: 0x10
5. Center Right: 0x08
6. Back Right: 0x04

Additionally, this board and associated firmware assume active-low chip select, and that the register is sharing a clock
with the SPI bus. 

#### Future development
Much of the firmware is still under development, and additional details such as component IDs, synchronization, and timing
requirements will be made available as that firmware is finalized. 

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
