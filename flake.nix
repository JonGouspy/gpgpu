{
  description = "CUDA GPGPU development environment (nvcc + toolkit + libs + profilers)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f (
            import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                cudaSupport = true;
              };
            }
          )
        );
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          lib = pkgs.lib;
          cuda = pkgs.cudaPackages_13_2;

          cmakeShim = pkgs.writeShellScriptBin "cmake" ''
            case "''${1:-}" in
              ""|--build|--install|--open|--workflow|-E|-P|--version|-h|--help|--help-*|--system-information|--find-package)
                exec ${pkgs.cmake}/bin/cmake "$@" ;;
            esac
            exec ${pkgs.cmake}/bin/cmake \
              -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
              -DCMAKE_POLICY_DEFAULT_CMP0135=NEW \
              -DCMAKE_POLICY_DEFAULT_CMP0127=NEW \
              -Wno-dev \
              -Wno-deprecated \
              "$@"
          '';

          cmakePatched = pkgs.symlinkJoin {
            name = "cmake-patched-${pkgs.cmake.version}";
            paths = [
              cmakeShim
              pkgs.cmake
            ];
          };

          # Profilers
          cupti = cuda.cuda_cupti or null;
          nvprof = cuda.cuda_nvprof or null;
          nsys = cuda.nsight_systems or null;
          ncu = cuda.nsight_compute or null;

          profilingTools = builtins.filter (p: p != null) [
            cupti
            nsys
            ncu
          ];

          guiWrapper =
            name: pkg:
            pkgs.writeShellScriptBin name ''
              exec ${pkgs.bash}/bin/bash ${pkg}/bin/${name} "$@"
            '';

          guiWrappers =
            lib.optional (nsys != null) (guiWrapper "nsys-ui" nsys)
            ++ lib.optional (ncu != null) (guiWrapper "ncu-ui" ncu);
        in
        {
          default = pkgs.mkShell {
            name = "cuda-gpgpu";

            nativeBuildInputs =
              guiWrappers
              ++ [
                cuda.cuda_nvcc
                cuda.cuda_cuobjdump
                cmakePatched
                pkgs.ninja
                pkgs.gnumake
                pkgs.pkg-config
              ]
              ++ profilingTools;

            buildInputs = [
              cuda.cuda_cudart
              cuda.cuda_cccl
              cuda.libcublas
              cuda.libcufft
              cuda.libcurand
              cuda.libcusparse
              cuda.libcusolver
              # cuda.cudatoolkit

              pkgs.zlib
              pkgs.libpng
              pkgs.onetbb
              pkgs.gbenchmark
              pkgs.gtest
              pkgs.ffmpeg
              pkgs.bc

              # GStreamer
              pkgs.gst_all_1.gstreamer
              pkgs.gst_all_1.gst-plugins-base
              pkgs.gst_all_1.gst-plugins-good
              pkgs.gst_all_1.gst-plugins-bad
              pkgs.gst_all_1.gst-libav
              pkgs.glib
            ];

            shellHook = ''
              export CUDA_PATH="${cuda.cuda_cudart}"
              export CUDACXX="${cuda.cuda_nvcc}/bin/nvcc"
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
              ${lib.optionalString (cupti != null) ''
                # some tools load libcupti at runtime
                export LD_LIBRARY_PATH="${lib.getLib cupti}/lib:$LD_LIBRARY_PATH"
              ''}

              export CMAKE_POLICY_VERSION_MINIMUM=3.5

              echo "CUDA shell ready."
              echo "  nvcc:   $(nvcc --version | grep release || echo 'not found')"
              echo "  cmake:  $(cmake --version | head -n1)"
              echo "  nsys:   $(nsys --version 2>/dev/null | head -n1 || echo 'not available')"
              echo "  ncu:    $(ncu --version 2>/dev/null | tail -n1 || echo 'not available')"
              echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
            '';
          };
        }
      );
    };
}
