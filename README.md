# Face Swap (on-device) for SillyTavern

Puts a face you choose onto the images your AI creates. Everything happens
right in your browser — no photos are ever sent anywhere.

## How it works, simply

1. You show it one clear photo of a face (your reference face).
2. When a new image is made in chat, the extension finds the face in that
   image and gently blends your reference face over it.
3. It can also tidy up the result a little so it looks more natural.

Think of it like a sticker of a face that the app cuts out and presses
onto new pictures for you, automatically.

## Install

**Option A — from GitHub (recommended):**
1. Copy your repo link (it ends in `.git`).
2. In SillyTavern, open the Extensions panel (puzzle piece) → Install extension.
3. Paste the link, install, then reload the page.

**Option B — from a zip file:**
1. Unzip the file.
2. Put the `ST-FaceSwap` folder inside SillyTavern's
   `public/scripts/extensions/third-party/` folder.
3. Reload the page.

## First-time setup (do once)

1. Open SillyTavern settings → Extensions → **Face Swap (on-device)**.
2. Click **Download models** and wait until it says ready.
   (This is a one-time download of about 1 GB, so use Wi-Fi.)
3. Click the file box under **Reference face** and pick one clear,
   front-facing photo. You should see a small preview of it.
4. Tick **Auto-swap generated images** if you want every new image
   swapped on its own.

That's it — no other setup needed.

## Everyday use

- **Automatic:** with the tick on, each new generated image is swapped
  by itself. A small note pops up while it works.
- **Manual:** click **Run on last generated image** to redo the latest one.
- **Command:** type `/faceswap` in chat to swap the latest image,
  or `/faceswap 12` to swap message number 12.

Your original picture is always kept safe behind the scenes, so swapping
again starts from the untouched original, not from an already-swapped one.

## Small tips

- Use a clear, well-lit, front-facing reference photo for the best look.
- Big, clear faces in the generated image swap better than tiny ones.
- If a result looks off, try a different reference photo.
- The **Light face enhance** tick smooths the result a little.
  Turn it off if things feel slow.
- Best in Chrome or Edge (they support the faster mode). If you see
  errors, switch **Runtime** to **WASM only** and try again.

## Good to know

- Only use faces you have permission to use — for example your own
  face or a friend who said yes.
- The first swap after opening SillyTavern takes longer while things
  warm up. Later swaps are quicker.
- Models are saved in your browser, so you only download them once.
  Nothing about your photos leaves your computer.

## If something goes wrong

| What you see | Try this |
|---|---|
| "Set a reference face" | Add a photo in the settings first |
| "No face found in the reference image" | Use a clearer, front-facing photo |
| "No face found" on a chat image | The face may be too small or turned away |
| "Download failed" | Check your internet, then click **Download models** again |
| Everything is slow or crashes | Turn off **Light face enhance**, or use **WASM only** |
| "ceil() in shape computation" error | Switch **Runtime** to **WASM only** (the extension also does this by itself and retries) |

Enjoy!
