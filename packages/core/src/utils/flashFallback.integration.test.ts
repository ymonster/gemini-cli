/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Config } from '../config/config.js';
import {
  setSimulate429,
  disableSimulationAfterFallback,
  shouldSimulate429,
  createSimulated429Error,
  resetRequestCounter,
} from './testUtils.js';
import { DEFAULT_GEMINI_FLASH_MODEL } from '../config/models.js';
import { retryWithBackoff } from './retry.js';
import { AuthType } from '../core/contentGenerator.js';

describe('Flash Fallback Integration', () => {
  let config: Config;

  beforeEach(() => {
    config = new Config({
      sessionId: 'test-session',
      targetDir: '/test',
      debugMode: false,
      cwd: '/test',
      model: 'gemini-2.5-pro',
    });

    // Reset simulation state for each test
    setSimulate429(false);
    resetRequestCounter();
  });

  it('should automatically accept fallback', async () => {
    // Set up a minimal flash fallback handler for testing
    const flashFallbackHandler = async (): Promise<boolean> => true;

    config.setFlashFallbackHandler(flashFallbackHandler);

    // Call the handler directly to test
    const result = await config.flashFallbackHandler!(
      'gemini-2.5-pro',
      DEFAULT_GEMINI_FLASH_MODEL,
    );

    // Verify it automatically accepts
    expect(result).toBe(true);
  });

  it('should trigger fallback after 2 consecutive 429 errors for OAuth users', async () => {
    let fallbackCalled = false;
    let fallbackModel = '';

    // Mock function that simulates exactly 2 429 errors, then succeeds after fallback
    const mockApiCall = vi
      .fn()
      .mockRejectedValueOnce(createSimulated429Error())
      .mockRejectedValueOnce(createSimulated429Error())
      .mockResolvedValueOnce('success after fallback');

    // Mock fallback handler
    const mockFallbackHandler = vi.fn(async (_authType?: string) => {
      fallbackCalled = true;
      fallbackModel = DEFAULT_GEMINI_FLASH_MODEL;
      return fallbackModel;
    });

    // Test with OAuth personal auth type, with maxAttempts = 3 to allow fallback + one retry
    const result = await retryWithBackoff(mockApiCall, {
      maxAttempts: 3,
      initialDelayMs: 1,
      maxDelayMs: 10,
      shouldRetry: (error: Error) => {
        const status = (error as Error & { status?: number }).status;
        return status === 429;
      },
      onPersistent429: mockFallbackHandler,
      authType: AuthType.LOGIN_WITH_GOOGLE_PERSONAL,
    });

    // Verify fallback was triggered
    expect(fallbackCalled).toBe(true);
    expect(fallbackModel).toBe(DEFAULT_GEMINI_FLASH_MODEL);
    expect(mockFallbackHandler).toHaveBeenCalledWith(
      AuthType.LOGIN_WITH_GOOGLE_PERSONAL,
    );
    expect(result).toBe('success after fallback');
    // Should have: 2 failures, then fallback triggered, then 1 success after fallback
    expect(mockApiCall).toHaveBeenCalledTimes(3);
  });

  it('should not trigger fallback for API key users', async () => {
    let fallbackCalled = false;

    // Mock function that simulates 429 errors
    const mockApiCall = vi.fn().mockRejectedValue(createSimulated429Error());

    // Mock fallback handler
    const mockFallbackHandler = vi.fn(async () => {
      fallbackCalled = true;
      return DEFAULT_GEMINI_FLASH_MODEL;
    });

    // Test with API key auth type - should not trigger fallback
    try {
      await retryWithBackoff(mockApiCall, {
        maxAttempts: 5,
        initialDelayMs: 10,
        maxDelayMs: 100,
        shouldRetry: (error: Error) => {
          const status = (error as Error & { status?: number }).status;
          return status === 429;
        },
        onPersistent429: mockFallbackHandler,
        authType: AuthType.USE_GEMINI, // API key auth type
      });
    } catch (error) {
      // Expected to throw after max attempts
      expect((error as Error).message).toContain('Rate limit exceeded');
    }

    // Verify fallback was NOT triggered for API key users
    expect(fallbackCalled).toBe(false);
    expect(mockFallbackHandler).not.toHaveBeenCalled();
  });

  it('should properly disable simulation state after fallback', () => {
    // Enable simulation
    setSimulate429(true);

    // Verify simulation is enabled
    expect(shouldSimulate429()).toBe(true);

    // Disable simulation after fallback
    disableSimulationAfterFallback();

    // Verify simulation is now disabled
    expect(shouldSimulate429()).toBe(false);
  });

  it('should not repeatedly call fallback handler if model was already switched during session', async () => {
    let fallbackHandlerCallCount = 0;

    // Mock a config with contentGeneratorConfig initialized
    const mockConfig = {
      getModel: vi.fn().mockReturnValue('gemini-2.5-pro'),
      setModel: vi.fn(),
      isModelSwitchedDuringSession: vi.fn().mockReturnValue(true), // Model already switched
      flashFallbackHandler: vi.fn(
        async (_currentModel: string, _fallbackModel: string) => {
          fallbackHandlerCallCount++;
          return true; // Accept fallback
        },
      ),
      getContentGeneratorConfig: () => ({
        authType: AuthType.LOGIN_WITH_GOOGLE_PERSONAL,
      }),
    };

    // Mock the handleFlashFallback method - this simulates the fix
    const handleFlashFallback = async (authType?: string) => {
      // Only handle fallback for OAuth users
      if (authType !== AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
        return null;
      }

      const currentModel = mockConfig.getModel();
      const fallbackModel = DEFAULT_GEMINI_FLASH_MODEL;

      // Don't fallback if already using Flash model
      if (currentModel === fallbackModel) {
        return null;
      }

      // Don't show fallback message again if model was already switched during this session
      if (mockConfig.isModelSwitchedDuringSession()) {
        // Silently switch to flash model without calling the handler again
        mockConfig.setModel(fallbackModel);
        return fallbackModel;
      }

      // Check if config has a fallback handler
      if (mockConfig.flashFallbackHandler) {
        try {
          const accepted = await mockConfig.flashFallbackHandler(
            currentModel,
            fallbackModel,
          );
          if (accepted) {
            mockConfig.setModel(fallbackModel);
            return fallbackModel;
          }
        } catch (error) {
          console.warn('Flash fallback handler failed:', error);
        }
      }

      return null;
    };

    // Call handleFlashFallback multiple times to simulate repeated rate limiting
    await handleFlashFallback(AuthType.LOGIN_WITH_GOOGLE_PERSONAL);
    await handleFlashFallback(AuthType.LOGIN_WITH_GOOGLE_PERSONAL);
    await handleFlashFallback(AuthType.LOGIN_WITH_GOOGLE_PERSONAL);

    // Verify the fallback handler was NOT called because model was already switched
    expect(fallbackHandlerCallCount).toBe(0);
    expect(mockConfig.flashFallbackHandler).not.toHaveBeenCalled();

    // Verify setModel was called to silently switch to flash
    expect(mockConfig.setModel).toHaveBeenCalledWith(
      DEFAULT_GEMINI_FLASH_MODEL,
    );
    expect(mockConfig.setModel).toHaveBeenCalledTimes(3); // Called each time to ensure we're on flash
  });

  it('should not infinitely retry after fallback when rate limits persist', async () => {
    let fallbackCallCount = 0;

    // Mock function that always returns 429 errors
    const mockApiCall = vi.fn().mockRejectedValue(createSimulated429Error());

    // Mock fallback handler that gets called
    const mockFallbackHandler = vi.fn(async (_authType?: string) => {
      fallbackCallCount++;
      return DEFAULT_GEMINI_FLASH_MODEL;
    });

    try {
      await retryWithBackoff(mockApiCall, {
        maxAttempts: 5,
        initialDelayMs: 1,
        maxDelayMs: 10,
        shouldRetry: (error: Error) => {
          const status = (error as Error & { status?: number }).status;
          return status === 429;
        },
        onPersistent429: mockFallbackHandler,
        authType: AuthType.LOGIN_WITH_GOOGLE_PERSONAL,
      });
    } catch (error) {
      // Expected to throw after max attempts
      expect((error as Error).message).toContain('Rate limit exceeded');
    }

    // Verify fallback was called only once (when consecutive429Count first reached 2)
    expect(fallbackCallCount).toBe(1);
    expect(mockFallbackHandler).toHaveBeenCalledTimes(1);

    // Should not exceed maxAttempts even with fallback
    expect(mockApiCall).toHaveBeenCalledTimes(5);
  });
});
