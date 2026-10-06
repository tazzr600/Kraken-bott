# =========================================================
# AUTONOMOUS LOOP
# =========================================================

async def run(self):

    # IMPORTANT:
    # The worker calls run() directly. The bot must explicitly
    # enter the running state here.
    self.running = True
    self.error = None

    print("=" * 60)
    print("AUTONOMOUS LOOP ONLINE")
    print("FULL MARKET SCANNER ONLINE")
    print("AI DAY TRADING ENGINE ONLINE")
    print(
        f"PAPER MODE: "
        f"{settings.dry_run}"
    )
    print("=" * 60)

    try:

        while self.running:

            try:

                # -------------------------------------------------
                # MANAGE EXISTING POSITION
                # -------------------------------------------------

                await self.manage_positions()

                # -------------------------------------------------
                # SCAN
                # -------------------------------------------------

                if self.scan_requested:

                    print(
                        "MANUAL SCAN REQUEST ACCEPTED"
                    )

                    self.scan_requested = False

                await self.scan()

                # -------------------------------------------------
                # TRADE
                # -------------------------------------------------

                await self.maybe_enter_best()

                # -------------------------------------------------
                # EQUITY
                # -------------------------------------------------

                await self.mark_equity()

                self.error = None

            except Exception as exc:

                self.error = (
                    f"BOT ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(self.error)

                # Keep the autonomous engine alive after a
                # recoverable cycle error.

            # -----------------------------------------------------
            # NEXT CYCLE
            # -----------------------------------------------------

            if self.running:

                await asyncio.sleep(
                    max(
                        5,
                        settings.scan_seconds,
                    )
                )

    except asyncio.CancelledError:

        print(
            "AUTONOMOUS LOOP CANCELLED"
        )

        self.running = False

        raise

    finally:

        self.running = False

        print(
            "AUTONOMOUS LOOP OFFLINE"
        )
