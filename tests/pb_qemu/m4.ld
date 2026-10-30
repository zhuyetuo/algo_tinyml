MEMORY { ROM (rx) : ORIGIN = 0x00000000, LENGTH = 4M  RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 4M }
ENTRY(_start)
SECTIONS {
  .text : { KEEP(*(.vectors)) *(.text*) *(.rodata*) KEEP(*(.init)) KEEP(*(.fini)) . = ALIGN(4); } > ROM
  .ARM.exidx : { *(.ARM.exidx*) } > ROM
  __etext = .;
  .init_array : { PROVIDE(__init_array_start = .); KEEP(*(.init_array*)) PROVIDE(__init_array_end = .); } > ROM
  .preinit_array : { PROVIDE(__preinit_array_start = .); KEEP(*(.preinit_array*)) PROVIDE(__preinit_array_end = .); } > ROM
  .fini_array : { PROVIDE(__fini_array_start = .); KEEP(*(.fini_array*)) PROVIDE(__fini_array_end = .); } > ROM
  .data : { __data_start__ = .; *(.data*) . = ALIGN(4); __data_end__ = .; } > RAM
  .bss : { __bss_start__ = .; __bss_start = .; *(.bss*) *(COMMON) . = ALIGN(4); __bss_end__ = .; _end = .; end = .; __end__ = .; } > RAM
  __stack_top = ORIGIN(RAM) + LENGTH(RAM);
  __StackTop = __stack_top; __HeapLimit = __stack_top - 0x10000;
}
