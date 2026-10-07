            return self.play(*proto_animations, run_time=run_time, rate_func=rate_func,
                             lag_ratio=lag_ratio, register=register)
        if not proto_animations:
            log.warning("Called Scene.play_async with no animations")
            return
        animations = [await prepare_animation_async(anim) for anim in proto_animations]
        for anim in animations:
            anim.update_rate_info(run_time, rate_func, lag_ratio)
        self.pre_play()
        self.begin_animations(animations)
        duration = run_time or self.get_run_time(animations)
        if register:
            self.checkpoints.append((self.get_state(ignore=[self.camera.frame]), animations, duration, None))
        # requestAnimationFrame is the browser clock. The configured render FPS
        # controls update granularity, not elapsed wall-clock time.
        start_time = await self._browser_frame()
        last_t = 0.0
        while last_t < duration:
            self.window.poll_events()
            if self.should_end_playing:
                self.should_end_playing = False
                break
            frame_time = await self._browser_frame()
            t = min(max(frame_time - start_time, 0.0), duration)
            dt = float(t - last_t)
            if dt <= 0:
                continue
            last_t = float(t)
            self.increment_time(dt)
            browser_audio.sync(self.time, self._interactive_sound_events)
            for animation in animations:
                animation.update_reference_mobjects(dt, frame_rate=self.camera.fps)
                animation.interpolate(float(t) / animation.run_time)
            await self.update_mobjects_async(dt)
            browser_audio.sync(self.time, self._interactive_sound_events)
            self.draw_frame(dt, force_draw=True)
            self.emit_frame()
        self.finish_animations(animations)
        self.post_play()

    async def wait_async(
        self,
        duration: float | None = None,
        register: bool = True,
    ) -> None:
        """Wait while yielding frames to requestAnimationFrame in the browser."""
        if sys.platform != "emscripten":
            return self.wait(duration=duration, register=register)
        duration = self.default_wait_time if duration is None else duration
        self.pre_play()
        await self.update_mobjects_async(0)
        if register:
            self.checkpoints.append((self.get_state(ignore=[self.camera.frame]), [], duration, None))
        last_t = 0.0
        for t in np.arange(0, duration, 1 / self.camera.fps) + 1 / self.camera.fps:
            self.window.poll_events()
            dt = float(t - last_t)
            last_t = float(t)
            await self.update_frame_async(dt, force_draw=True)
            browser_audio.sync(self.time, self._interactive_sound_events)
            self.emit_frame()
            await self._browser_frame()
        self.post_play()
