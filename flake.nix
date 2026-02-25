{
  description = "Phase Alignment Lab - DSP prototyping environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  nixConfig = {
    extra-substituters = [ "https://cache.nixos.org" ];
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        unfree = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python312;
        pythonPackages = python.pkgs;
      in
      {
        # Easy access: nix run .#claude
        apps.claude = {
          type = "app";
          program = "${unfree.claude-code}/bin/claude";
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPackages.numpy
            pythonPackages.scipy
            pythonPackages.matplotlib
            pythonPackages.soundfile
            pythonPackages.librosa
            pythonPackages.ipython
            pythonPackages.jupyter

            # For audio playback
            pkgs.portaudio
            pythonPackages.sounddevice

            # Claude Code CLI
            unfree.claude-code
          ];

          shellHook = ''
            echo "Phase Alignment Lab"
            echo "==================="
            echo "Available tools:"
            echo "  - numpy, scipy: DSP fundamentals"
            echo "  - librosa: Audio analysis"
            echo "  - soundfile: WAV I/O"
            echo "  - matplotlib: Visualization"
            echo "  - sounddevice: Audio playback"
            echo "  - claude: Claude Code CLI"
            echo ""
            echo "Run: python phase_align.py"
          '';
        };
      });
}
