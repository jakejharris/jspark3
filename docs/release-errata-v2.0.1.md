> [!IMPORTANT]
> **v2.0.1 installation and restart errata (2026-10-04)**
>
> Use the maintained [INSTALL](https://github.com/jakejharris/jspark3/blob/main/INSTALL.md),
> [UPGRADING](https://github.com/jakejharris/jspark3/blob/main/UPGRADING.md) and
> [known issues and hotfixes](https://github.com/jakejharris/jspark3/blob/main/docs/TROUBLESHOOTING.md#v201-known-issues-and-hotfixes).
> These supersede the tagged guides where corrected: Docker-group setup, disk requirements, no-drafter preflight,
> rollback downloads and restart/compile-lock recovery. Unmodified v2.0.1 rebuilds kernels on every start;
> the optional kernel rebuild hotfix is **validated on 3x DGX Spark (2026-10-04): retained boot reused the kernel cache; undo restored the original files**.
> Continue running the recipe from the immutable `v2.0.1` tag, following the maintained guides.
