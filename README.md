# ShoeBot

## Statement of Purpose
The ShoeBot framework is an open-source platform intended to lower the barrier of entry to learning mobile robotics and 
controls. As long as you have a box, access to a 3D printer, as well as soldering and crimping capabilities, then you 
should be able to use this framework. Through the use of a standardized mounting system, creators have the freedom to 
use either the pre-built demos to get a cheap bot up and running or develop their own, compatible platforms. 

## Description of Modules

1. Wheels: This module consists of DC motors and mecanum wheels.
2. Shoeshine: Offers quadrapedal motion through use of servomotors
3. HexaBox: Requires six mounts, offers six-legged locomotion through use of servomotors.

## Hardware files
STEP files for all prepared modules will be made available upon finalization.

## FAQ

1. Why shoeboxes?

>I had originally set out to make what essentially amounts to just the wheels module as a CAD and mechatronic portfolio piece.
Seeking to understand the design considerations, I started to immerse myself in literature surrounding mecanum wheel devices 
and discovered something that is in hindsight self-evident: different design decisions like the wheel spacing and orientation 
fundamentally alters the dynamics of the system and therefore influence control. Creating modular systems to explore the impact 
of these parameters was largely underexplored in what I found, so I set out to make my own. But I needed a good frame. Something
cheap and easy to deploy. As I looked around me, I saw an unused shoebox. And the ShoeBot system was born. 

Realistically any cardboard box will work, but at the time a shoebox seemed like the right size for what I set out to accomplish.

2. Why didn't you use \<insert fastener here\> to fasten the modules to the box?

>In the early days, I explored a number of different means for mounting the hardware to the box. The current design for mounting
hardware to the box solves two problems simulataneously: the first is that of establishing a secure connection, the second is
offering a path for routing the wires from inside the box to the modules on the outside. Does the act of cutting into the box
limit the reusability of a single box to test multiple design parameters? Yes, as do other adhesives that have enough strength
to reliably anchor the hardware to the outside of the box. 
