{
  description = "Magic Phase - Spectral Phase Alignment Plugin";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            freetype
            alsa-lib
            webkitgtk
            curl
            gtk3
            jack2
            xorg.libX11
            xorg.libX11.dev
            xorg.libXext
            xorg.libXinerama
            xorg.xrandr
            xorg.libXcursor

            pcre2
            pcre
            libuuid
            libselinux
            libsepol
            libthai
            libdatrie
            libpsl
            xorg.libXdmcp
            libxkbcommon
            libepoxy
            xorg.libXtst
            libsysprof-capture
            sqlite.dev

            # Python prototypes
            (python3.withPackages (ps: with ps; [
              numpy
              scipy
              matplotlib
              soundfile
            ]))
          ];

          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
            gnumake
            patchelf
            gdb
          ];

          NIX_LDFLAGS = toString [
            "-lX11"
            "-lXext"
            "-lXcursor"
            "-lXinerama"
            "-lXrandr"
          ];

          hardeningDisable = [ "fortify" ];

          shellHook = ''
            export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
            echo "Magic Phase development environment"
            echo "Build parallelism: $CMAKE_BUILD_PARALLEL_LEVEL cores"
          '';
        };
      });
}
