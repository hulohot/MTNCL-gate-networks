module xor_mtncl(
  input  wire [1:0] in_bits,
  output wire [1:0] out_bits
);

  wire input_1 = in_bits[0];
  wire input_2 = in_bits[1];
  wire const_3 = 1'b1;
  wire const_4 = 1'b0;

  wire gate_l1_5 = (const_4 | input_1);
  wire gate_l1_6 = (input_2 & const_4 & const_3);
  wire gate_l1_7 = (input_2 & input_1);
  wire gate_l1_8 = (input_1 & const_4);
  wire gate_l2_9 = (gate_l1_5 & const_4);
  wire gate_l2_10 = (gate_l1_5 ^ input_2);
  wire output_0_11 = (gate_l1_7 & gate_l1_5);
  wire output_1_12 = (((gate_l1_5 ? 1 : 0) + (gate_l1_6 ? 1 : 0) + (gate_l2_10 ? 1 : 0)) >= 1);

  assign out_bits[0] = output_0_11;
  assign out_bits[1] = output_1_12;
endmodule
