import shutil
import subprocess
from pathlib import Path

import pytest

from mtncl_nn import MTNCLNetwork


def test_xor_verilog_with_iverilog(tmp_path: Path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog/vvp not installed")

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[1, 0], [0, 1], [0, 1], [1, 0]]

    net = MTNCLNetwork(num_inputs=2, num_outputs=2, hidden_layers=[4, 2])
    net.train(X, y, iterations=1200)

    verilog = net.to_verilog(module_name="xor_mtncl")
    design_path = tmp_path / "xor_mtncl.v"
    tb_path = tmp_path / "tb.v"
    out_path = tmp_path / "a.out"

    design_path.write_text(verilog, encoding="utf-8")

    tb_path.write_text(
        """
module tb;
  reg  [1:0] in_bits;
  wire [1:0] out_bits;

  xor_mtncl dut(
    .in_bits(in_bits),
    .out_bits(out_bits)
  );

  initial begin
    in_bits = 2'b00; #1; $display("00 %b", out_bits);
    in_bits = 2'b01; #1; $display("01 %b", out_bits);
    in_bits = 2'b10; #1; $display("10 %b", out_bits);
    in_bits = 2'b11; #1; $display("11 %b", out_bits);
    $finish;
  end
endmodule
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["iverilog", "-o", str(out_path), str(design_path), str(tb_path)], check=True)
    result = subprocess.run(["vvp", str(out_path)], check=True, capture_output=True, text=True)

    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    observed = {ln.split()[0]: ln.split()[1] for ln in lines if len(ln.split()) == 2}

    assert observed.get("00") == "10"
    assert observed.get("01") == "01"
    assert observed.get("10") == "01"
    assert observed.get("11") == "10"
