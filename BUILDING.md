# Building the MosaicGone APK

These are the steps used to build MosaicGone's releases (up to v0.5) on Windows 11. Other systems should work the same way with
`./gradlew` instead of `gradlew.bat`, but have not been tried.

## What you need

| Tool | Version used | Notes |
|---|---|---|
| JDK | 17 (OpenJDK 17.0.20.1) | set `JAVA_HOME` to it |
| Android SDK | platform `android-35`, build-tools 35.0.0 (34.0.0 for `apksigner` / `aapt2` checks) | `compileSdk 35`, `targetSdk 35`, `minSdk 29` |
| Gradle | 8.11.1 | downloaded by the wrapper (`gradlew.bat`); no separate install |
| Android Gradle Plugin / Kotlin | 8.7.3 / 2.0.21 | fetched by Gradle |

The first build downloads Gradle, the plugins and the libraries (see `THIRD_PARTY_NOTICES.md`). The models are already
in `app/src/main/assets/models/`, so no model download is needed to build the app.

## 1. Point the build at the Android SDK

Create `local.properties` in this folder (it is not in Git):

```
sdk.dir=C\:\\path\\to\\android-sdk
```

## 2. Build

From this folder:

```bat
set JAVA_HOME=C:\path\to\jdk17
gradlew.bat :app:assembleDebug
```

Result: `app/build/outputs/apk/debug/app-debug.apk`, signed with the SDK's debug key. It has the same app id as the
release build, so it cannot be installed over a release build (different key) without uninstalling that first.

### Release build

```bat
gradlew.bat :app:assembleRelease
```

- **With a signing key:** create `keystore.properties` in this folder. It is not in Git, and the key file it points
  to must never be published.

  ```
  storeFile=keystore/your-release-key.jks
  storePassword=...
  keyAlias=...
  keyPassword=...
  ```
  Result: `app/build/outputs/apk/release/app-release.apk`, signed with that key.
- **Without `keystore.properties`:** the release build is not signed
  (`app/build/outputs/apk/release/app-release-unsigned.apk`) and must be signed before Android will install it.

Official MosaicGone releases are signed with the maintainer's own key (certificate `CN=MosaicGone, O=MosaicGone`).
An APK you build yourself has a different signature, so Android will not install it over an official release, or
the reverse, without uninstalling first.

The release build also runs Android lint's release checks (`lintVitalRelease`); a lint error stops the build.

## 3. Check the APK (optional)

```bat
%ANDROID_SDK%\build-tools\34.0.0\apksigner.bat verify --print-certs app\build\outputs\apk\release\app-release.apk
%ANDROID_SDK%\build-tools\34.0.0\aapt2.exe dump badging app\build\outputs\apk\release\app-release.apk
```

## Tests

```bat
gradlew.bat :core:test :app:testDebugUnitTest
```

- `:app` tests (the link finder) need nothing extra.
- Most `:core` tests run the real models on the PC (desktop ONNX Runtime 1.20.0) and read their images and clips from
  `core/testdata/` (see its README). The fixtures they need are in Git, so the tests run from a fresh clone. The optional
  extra clips are not; `QualityTest` skips the parts that need them.
- Test output goes to `core/build/`, so `gradlew clean` removes only results, not fixtures.

## Rebuilding the models

Not needed to build the app. See `MODELS.md` and `model-development/README.md`.
